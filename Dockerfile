FROM innermost47/obsidian-neural-provider:base AS base

FROM base AS final
WORKDIR /app
COPY provider.py .
COPY entrypoint.sh .
RUN sed -i 's/\r//' entrypoint.sh && chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]