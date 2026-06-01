FROM obsidian-base:latest
WORKDIR /app
COPY provider.py .
COPY entrypoint.sh .
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh
ENV SA3_KEEP_IN_RAM=false
CMD ["./entrypoint.sh"]