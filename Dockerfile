FROM obsidian-base:latest
WORKDIR /app
COPY main.py .
COPY audio_generator.py .
COPY base_generator.py .
COPY models.py .
COPY sa_generator.py .
COPY sa3_generator.py .
COPY server_utils.py .
COPY settings.py .
COPY entrypoint.sh .
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh
ENV SA3_KEEP_IN_RAM=true
CMD ["./entrypoint.sh"]