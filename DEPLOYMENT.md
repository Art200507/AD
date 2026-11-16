# Deployment requirements

Use a container host for the Python model and Chroma index; this is not a static site. Set `GROQ_API_KEY` server-side, review the source documents, and run the ingestion step before testing retrieval. The UI no longer assumes a named person's identity or claims complete privacy. Chat history is bounded, expires after an hour, and is attached to a server-issued HttpOnly cookie rather than a client-supplied session ID. Restarting or scaling workers clears/partitions this in-memory history. Use one worker for this prototype.

Before public release: configure provider budget/rate limits, verify camera-free chat and streamed errors in a browser, review medical source accuracy, and test retrieval with the installed model. This prototype provides information and support, not diagnosis or replacement medical care.
