import json
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, MessagesPlaceholder
    )
from langchain.memory import ConversationSummaryBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class GPT(object):
    @classmethod
    def __init__(cls):
        cls.llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")
        cls.system_context_template = """
            You are a friendly and helpful fashion chatbot! Your goal is to assist users in finding the perfect fashion items by providing detailed and realistic suggestions. Your responses should be formatted as a JSON object with keys: "answer", "summary" and "fashion_items."

            - Engage with users in a helpful manner, offering fashion-related advice and suggestions.
            - Recommend items within the categories of coats, denims, dresses, jackets, knitwear, skirts, tops, or trousers.
            - Ensure your descriptions are detailed (more than five words each) and free from specific brand names.
            - Summarize the user's needs in a concise phrase (less than 5 words) in the "summary" key.
            - Create a list of suggested fashion items in the "fashion_items" key, separated by commas and enclosed in brackets.
            - Exclude shoes, bags, scarves, and other items not specified in the mentioned categories.

            Remember, never provide answers outside the fashion domain, and maintain a focus on practical and relevant fashion suggestions.
            """
        cls.system_format_template = """
            First, interact with the human, responding as a helpful chatbot assistant. Your detailed fashion suggestions go into the "answer" part of the final JSON output.
            Limit suggestions to coats, denims, dresses, jackets, knitwear, skirts, tops, or trousers.
            Ensure descriptions are detailed, concrete, and realistic (more than five words each). Exclude shoes, bags, scarves, and specific brand names.
            Second, summarize what the user wants in a short phrase (less than 5 words). This goes into the "summary" part of the final JSON output.
            Third, if you've suggested fashion items, create a list in the "fashion_items" part, separating them by commas and enclosing them in brackets.
            Include items only from coats, denims, dresses, jackets, knitwear, skirts, tops, or trousers. If no items suggested, use null.

            Your output should always be in JSON format with keys "answer," "summary," and "fashion_items," as specified above, regardless of the user input.
            Do not output anything other than the specified JSON markdown snippet code.

            Your responses should follow this structure:
            - "answer": [Your detailed and helpful fashion-related response],
            - "summary": [Concise summary of the user's needs, less than 5 words],
            - "fashion_items": [List of suggested fashion items, if applicable; otherwise, use null].

            For example:
            {
                "answer": "For a Halloween look, you could consider a black dress with a lace detail for a witchy vibe. Pair it with a faux leather jacket for an edgy touch. You could also consider a red skirt and a black top for a vampire-inspired look." ,
                "summary": "Halloween outfit suggestions",
                "fashion_items": ["black dress with lace detail", "faux leather jacket", "red skirt"]
            }
        """
        cls.system_limit_template = """
            Limit your responses to fashion-related queries only. Stick to the provided JSON format with "answer," "summary," and "fashion_items" keys. Your fashion suggestions will be used to search our Farfetch database.
            Guidelines:
            - Focus on women's fashion items: coats, denims, dresses, jackets, knitwear, skirts, tops, or trousers.
            - Avoid specific brand names for a more general and relatable experience.
            - Exclude items not available on Farfetch, those outside specified categories, or not relevant to women's clothing.

            Your aim is to provide practical and relevant fashion advice, enhancing the user's shopping experience.
        """
        cls.human_template = "{question}"
        cls.ai_template = ""
        cls.format_template = "{format_instructions}"

        response_schemas = [
            ResponseSchema(name="answer", description="Your detailed response to the user's fashion-related question. This is the information the user will see, so make it helpful and friendly."),
            ResponseSchema(name="summary", description="Provide a brief, catchy phrase that captures the essence of what the user is looking for in the world of fashion."),
            ResponseSchema(name="fashion_items", description="List the fashion items you suggested in your response, separated by commas and enclosed in brackets. These items should be within the categories of coats, denims, dresses, jackets, knitwear, skirts, tops, or trousers.", type="list")
        ]
        cls.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    
    @classmethod
    def get_response(cls, user_text, chat_history):
        memory = ConversationSummaryBufferMemory(llm=cls.llm, memory_key="chat_history", return_messages=True, max_token_limit=3000)
        if chat_history is not None:
            for user_chat in chat_history["user_chat"]:
                memory.chat_memory.add_user_message(user_chat["query"])
                gpt_chat = user_chat["gpt_chat"][0]
                gpt_response = dict.fromkeys(["answer", "summary", "fashion_items"])
                gpt_response["answer"] = gpt_chat["answer"]
                fashion_items = [gpt_chat["gpt_query1"], gpt_chat["gpt_query2"], gpt_chat["gpt_query3"]]
                gpt_response["fashion_items"] = fashion_items
                memory.chat_memory.add_ai_message(json.dumps(gpt_response))

        # Construct prompt from templates
        prompt = ChatPromptTemplate(
            messages = [
                SystemMessagePromptTemplate.from_template(cls.system_context_template),
                SystemMessagePromptTemplate.from_template(cls.format_template),
                SystemMessagePromptTemplate.from_template(cls.system_limit_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(cls.human_template),
                AIMessagePromptTemplate.from_template(cls.ai_template),
            ],
            input_variables={"question"},
            partial_variables={"format_instructions": cls.output_parser.get_format_instructions()}
        )

        chain = LLMChain(llm=cls.llm, prompt=prompt, verbose=True, memory=memory)
        llm_response = chain.predict(question=user_text)
        try : 
            response_dict = cls.output_parser.parse(llm_response)
            if response_dict.get('fashion_items') is not None:
                queries = [i for i in response_dict.get('fashion_items') if i is not None]
            else: queries = []
            
            response = {
                "answer" : response_dict.get('answer'),
                "summary" : response_dict.get('summary'),
                "gpt_queries" : queries
            }
            return response
        except:
            answer = "Apologies for the inconvenience. It appears there's a technical issue processing your input. Remember, I'm here exclusively for fashion-related questions. If you have another fashion inquiry or need style advice, please let me know, and I'll do my best to assist you!"
            response = {
                "answer" : answer,
                "summary" : user_text,
                "gpt_queries" : []
            }
            return response