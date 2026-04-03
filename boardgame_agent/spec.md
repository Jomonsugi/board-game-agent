The board game agent's purpose is to answer questions and make clarifications on the rules of a board game. 

The agent will be accessible through a chat interface via a local web app. Good use something simple like streamlit if possible. Might have to be more complex, but an unsure.

The agent will have two primary capabilities:
1. Retrieval system of .pdf documents
2. web search via Tavily

These capabilities essentially mimic how a human solves rules questions: The rulebook, sometimes supporting documents like a glossary or icon sheet and also by looking at trusted sources on the internet for clarification.

### Retrieval system of .pdf documents

The strategy and apis for the RAG system is demonstrated in rulebook_text_only_rag.ipynb
The demonstrated pipeline will need to be implemented in a workflow. Every time a user is playing a game, they can provide any number of pdfs that will go through the process of being extracted by docling, indexed and made available to the agent via a tool. Once pdfs are indexed for the game they are available moving forward. When the user plays the game again, the pdfs are already in the index. As well, the user could provide additional pdfs or have the option to remove a pdf if they wish. These are all tools the agent has available. The app could use a simple interface where a path to a folder or a single .pdf is given that can be used to direct the agent to the pdf file locations. Or a drag and drag interface could be even easier. 

What is critical about the system is that the agent has a tool available to show the pdfs with the applicable highlighted rules, demonstrated in rulebook_text_only_rag.ipynb
That means that the app will need the ability to show those highlights. What would be really useful is if the agent would always cite references and when the user clicks on the references from the agents answers the pdf page would be "scrolled" to or something like that and the citations would be highlighted for easy viewing. Plus making sure the page number and source pdf document is clearly understood so the user could reference the physical copy if they want. 

It is important that the indexed (qdrant) database is setup in such a way that only the documents for the game the user is playing are referenced. This will make retrieval much more efficient and of course only give the agent data that is valid for the game the user is playing. 

The app can also have a way to select games that already have indexed documents and add new games.

### web search via Tavily

this is a web search tool and to start it would only access boardgamegeek.com for all answers. The agent will use this tool just like a human would, for supporting evidence and for clarification. The agent can reference what it found in answers, but must provide a link to the reference AND it must always cite the accompanying rule in the rulebook with references whether it uses the web search tool or not. 

### Architecture and Memory

The system will be built with Langgraph. Not only should it have a memory of conversation history, but it will need to keep a state for each game. That way if the session is paused or stopped or if a user is playing a game months from now again, not only will the indexed pdfs be available, but the history of questions and answers so that the agent starts to build a knowledge base of previously asked and answered questions. That will help it improve its understanding of the game as time goes on. For langgraph "state" is critical, but also https://langchain-ai.github.io/langmem/#installation could be a useful tool. 

Every interaction with this agent via chat will be a question about a rule. Every time the agent will use the RAG tool and can use the web search tool for clarification and supporting evidence. It will always return citations in the pdfs with bbox's as shown the the notebook so it can cite references and they can be clicked on and shown in the app via highlights in the pdf. In addition references to boardgamegeek.com will be provided that a clickable in the chatbot answer. 

### V2 considerations

In a separate folder I already built rule_book_agent. That agent finds rulebooks. In the future I might want this entire graph to be made available to this new boardgame_rules_agent, but for now I decided that it is easy enough for the human to provide the rules pdf and any other supporting pdf documents. The user only has to do this once anyway for each game they are playing and then never again unless they want to provide a new supporting document or remove on (for example updated rules, but this would rarely or maybe never happen). The most likely scenario is for the user to provide a rulebook for the game once, that be processed and indexed and then every subsequent time they play the game the resource is already available. 

As well, I am calling the folder that the agent code will be in "boardgame_agent" because although I'm starting with an agent that is answering questions about rules, I have some other ideas I want to expand the agent with in the future. These will always take the form of new tools available to the agent so it is critical that the agent framework be based on available tool calls to complete task, not a rigid workflow. 
