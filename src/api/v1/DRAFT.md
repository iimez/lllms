### Llama.cpp API

Shared surface for all engines?

Stateless endpoints:
- `POST /llama/completion`
- `POST /llama/chat`

Simplest way for stateful chat?
- `GET /llama/chat/{chat_id}`
- `POST /llama/chat/{chat_id}`