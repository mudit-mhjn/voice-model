## How to build and run via Docker
1) Build (Run inside the base directory of the repository) - 
```
docker build -t voice-model:test .
```

2) Run - 
```
docker run -p 8000:8000 voice-model:test
```

> To run locally for development and to avoid rebuilds - 
> - Bind-mount ptoject into container
> - run uvicorn with --reload so code changes auto-reload
``` 
docker run --rm -p 8000:8000
    -v /voice-model:/app
    -w /app
    voice-model
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```