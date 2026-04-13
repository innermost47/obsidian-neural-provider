FROM innermost47/obsidian-neural-provider:base
WORKDIR /app
COPY provider.py .
COPY entrypoint.sh .
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]