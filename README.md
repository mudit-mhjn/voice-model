## How to build and run via Docker
1) Build (Run inside the base directory of the repository) - 
```
docker build -t voice-model:test .
```

2) Run - 
```
docker run -p 8000:8000 voice-model:test
```