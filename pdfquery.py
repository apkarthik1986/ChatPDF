import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import langchain

# Set debug attribute to avoid the error
setattr(langchain, 'debug', False)

class SimpleDocumentStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1536)
        self.documents = []
        self.vectors = None

    def add_documents(self, documents: List[Document]):
        self.documents.extend(documents)
        texts = [doc.page_content for doc in documents]
        if self.vectors is not None:
            new_vectors = self.vectorizer.transform(texts).toarray()
            self.vectors = np.vstack([self.vectors, new_vectors])
        else:
            self.vectors = self.vectorizer.fit_transform(texts).toarray()

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.documents:
            return []
        query_vector = self.vectorizer.transform([query]).toarray()
        scores = cosine_similarity(query_vector, self.vectors)[0]
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

    def as_retriever(self):
        return self


class PDFQuery:
    def __init__(self, google_api_key = None) -> None:
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        
        # Initialize attributes
        self.db = SimpleDocumentStore()
        self.chain = None
        
        # Configure the Google Generative AI
        genai.configure(api_key=google_api_key)
        
        # Initialize the LLM with required parameters
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-pro-latest",
            temperature=0.3,
            google_api_key=google_api_key,
            convert_system_message_to_human=True,
            client_options=None,
            transport=None,
            client=None
        )

    def ask(self, question: str, chat_history: Optional[List[Any]] = None) -> str:
        if self.chain is None or self.db is None:
            return "Please, add a document."
        try:
            docs = self.db.similarity_search(question)
            if not docs:
                return "No relevant information found in the document."
            # Concatenate document contents
            context = "\n".join([doc.page_content for doc in docs])

            # Build conversation history snippet (last N turns) if provided
            conv_snippet = ""
            if chat_history:
                try:
                    # chat_history is a list of (message, is_user) tuples stored by the UI
                    # take the last 6 messages (3 user/assistant turns)
                    last = chat_history[-6:]
                    lines = ["Conversation so far:"]
                    for msg, is_user in last:
                        role = "User" if is_user else "Assistant"
                        # keep message short to avoid long prompts
                        txt = str(msg).strip()
                        lines.append(f"{role}: {txt}")
                    conv_snippet = "\n".join(lines) + "\n\n"
                except Exception:
                    conv_snippet = ""

            # Compose the final prompt including conversation snippet and context
            prompt = (
                "You are a helpful AI assistant. Use the conversation history and the provided "
                "context to answer the question. If the answer cannot be found in the context, "
                "say 'I don't have enough information to answer that question.'\n\n"
                f"{conv_snippet}Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )

            # Get response from the model
            result = self.chain(prompt)
            # Prefer structured content if available
            content = None
            try:
                if hasattr(result, "content"):
                    content = result.content
                elif isinstance(result, dict) and "content" in result:
                    content = result["content"]
                else:
                    content = result
            except Exception:
                content = result

            # Normalize to string
            if isinstance(content, list):
                content = " ".join([str(item) for item in content])
            content = str(content)


            # If the model returned a serialized representation like
            # "content='...'<other fields>", try to extract the inner content
            import re as _re
            m = _re.search(r"content=(?:'|\")(.*?)(?:'|\")\s+response_metadata=", content, flags=_re.S)
            if m:
                content = m.group(1)

            # Convert literal escaped newlines ("\\n") into real newlines
            content = content.replace('\\r\\n', '\n').replace('\\r', '\n').replace('\\n', '\n')

            # Turn inline bullet separators like " * " into line bullets
            content = content.replace(' * ', '\n* ')

            # Collapse multiple blank lines to two (paragraph separation)
            content = __import__('re').sub(r"\n\s*\n+", "\n\n", content)

            # Preserve single newlines (useful for lists); trim paragraphs
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            cleaned = '\n\n'.join(paragraphs)
            return cleaned.strip()

        except Exception as e:
            print(f"Error in ask: {e}")
            return f"An error occurred while processing your question. Please try again."

    def ingest(self, file_path: os.PathLike) -> None:
        loader = UnstructuredPDFLoader(str(file_path))
        split_docs = loader.load_and_split()
        split_core_docs = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in split_docs]
        self.db.add_documents(split_core_docs)
        self.chain = lambda prompt: self.llm.invoke(prompt)

    def forget(self) -> None:
        """Reset the document store with a new instance."""
        try:
            self.db = SimpleDocumentStore()
            self.chain = None
        except Exception as e:
            print(f"Error in forget: {e}")
            self.db = SimpleDocumentStore()
            self.chain = None
