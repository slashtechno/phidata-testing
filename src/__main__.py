from phi.assistant import Assistant
from phi.llm.ollama import Ollama
import ollama
from phi.vectordb.pgvector import PgVector2
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader

model = "llama3.1:8b"
# model = "adrienbrault/nous-hermes2theta-llama3-8b:q8_0"
# model = "nexusraven"
host = None

ollama_client = ollama.Client(host=host)
ollama_client.pull(model=model)
llm = Ollama(model=model)


# website_knowledge_base = WebsiteKnowledgeBase(
# vector_db=PgVector2(
# collection="website_documents",
#     db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
#     embedder=OllamaEmbedder(
#         model=model,
#     ),
# ),
# )

knowledge_base = PDFKnowledgeBase(
    path="data",
    vector_db=PgVector2(
        collection="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OllamaEmbedder(
            model="nomic-embed-text:latest",
            dimensions=768,
        ),
    ),
    reader=PDFReader(chunk=True)
)


assistant = Assistant(
    llm=llm,
    debug_mode=True,
    knowledge_base=knowledge_base,
    add_references_to_prompt=True,
)


# assistant.knowledge_base.load(recreate=False)
assistant.knowledge_base.load(recreate=True, upsert=True)
# assistant.cli_app()
assistant.print_response("Who was the clinical team leader", markdown=False)
