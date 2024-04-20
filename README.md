## Node Inference Server

Providing OpenAI API endpoints for gpt4all and node-llama-cpp.

### TODO

- v1/chat/completions
- v1/completions
- v1/models
- streaming


### Run Standalone

```
npm install
npm run build
npm run start
```

```
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "llama3:8b",
      "messages": [
          {
              "role": "user",
              "content": "Whats 1+1?"
          }
      ]
  }'
```

```
curl http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Say this is a test",
    "max_tokens": 7,
    "temperature": 0
  }'
```