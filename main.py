from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

#loading HuggingFace summarization pipeline
summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")

#wrap it inside LangChain
llmodel = HuggingFacePipeline(pipeline=summarizer_model)

#create a prompt template for summarization
prompt = PromptTemplate.from_template(
    """Summarize the following text in a way a {age} year old would understand:\n\n{text}""")

#create a chain for the summarization
chain = prompt | llmodel


#Get user input to fill the voids in template
text_to_summarize = input("\nPlease enter a topic: ")
age = input("\nEnter age of your choice:")

#Execute/Invoke the summarization-chain created
summary = chain.invoke({"text": text_to_summarize,
                        "age": age})

print("Here's the summary:")
print(summary)