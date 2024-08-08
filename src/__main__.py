from phi.assistant import Assistant

# from phi.tools.website import WebsiteTools
from phi.llm.ollama import Ollama
import ollama

# from phi.knowledge.website import WebsiteKnowledgeBase
# from phi.knowledge.combined import CombinedKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase


model = "llama2-uncensored"
# model = "llama3.1:8b"
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

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://www.accessdata.fda.gov/drugsatfda_docs/nda/2018/210951Orig1s000MultidisciplineR.pdf", "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2018/rev_210951_arn-509-003_PRO_Redacted.pdf"],
    vector_db=PgVector2(
        collection="pdf_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        embedder=OllamaEmbedder(
            model=model,
        ),
    ),
)


assistant = Assistant(
    llm=llm,
    debug_mode=True,
    prevent_hallucinations=False,
    knowledge_base=knowledge_base,
)
# assistant.knowledge_base.load(recreate=False)
assistant.knowledge_base.load()
assistant.cli_app()
# assistant.print_response("Who was the clinical team leader in the multi-discipline review of erleada by the FDA?", markdown=False)
print("Done!")
