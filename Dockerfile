# -- Stage 1: Build --
FROM rust:slim-bookworm AS builder

WORKDIR /usr/src/gateway
COPY . .
# Build project ở chế độ release
RUN cargo build --release

# -- Stage 2: Runtime (Cực nhẹ) --
FROM debian:bookworm-slim

# Cài đặt chứng chỉ CA để gọi API của Google an toàn qua HTTPS
RUN apt-get update \
    && apt-get install -y ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary từ build stage
COPY --from=builder /usr/src/gateway/target/release/gateway-for-vertex-ai /usr/local/bin/

# Cloud Run / Docker luôn ưu tiên listen trên 0.0.0.0 thay vì 127.0.0.1
ENV BIND_ADDR=0.0.0.0:8787
ENV RUST_LOG=gateway_for_vertex_ai=info

EXPOSE 8787

ENTRYPOINT ["gateway-for-vertex-ai"]
