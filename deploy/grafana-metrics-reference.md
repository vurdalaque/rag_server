# Grafana metrics reference for rag_server

VictoriaMetrics scrapes `GET /metrics` on rag_server. Grafana uses the
VictoriaMetrics datasource; this document lists the exported metrics and
suggested panels.

## Core health

| Metric | Type | Panel idea |
|--------|------|------------|
| `rag_index_loaded` | gauge | stat 0/1 |
| `rag_documents_total` | gauge | stat |
| `rag_index_dimensions` | gauge | stat |
| `rag_rerank_enabled` | gauge | stat |
| `rag_dependency_up{service}` | gauge | stat per service |

## Dependency alerts (high priority)

```promql
rag_dependency_up{service="embedding"} == 0
```

```promql
rate(rag_dependency_errors_total{service="embedding",error_type="connect"}[5m]) > 0
```

Services: `embedding`, `llm`, `searxng`.

## MCP / search

```promql
rate(rag_mcp_tool_calls_total{tool="search_project",outcome="error"}[5m])
/
rate(rag_mcp_tool_calls_total{tool="search_project"}[5m])
```

```promql
histogram_quantile(0.95, sum(rate(rag_mcp_tool_duration_seconds_bucket[5m])) by (le, tool))
```

## Retrieve pipeline

```promql
histogram_quantile(0.95, sum(rate(rag_retrieve_duration_seconds_bucket[5m])) by (le))
```

```promql
rate(rag_retrieve_empty_total[5m])
```

## HTTP

```promql
sum(rate(rag_http_requests_total[5m])) by (handler, status)
```

## Chat completions

```promql
sum(rate(rag_chat_completions_total[5m])) by (stream, outcome)
```

```promql
histogram_quantile(0.95, sum(rate(rag_chat_rag_sources_bucket[5m])) by (le))
```

## Cardinality

Do not use high-cardinality labels in dashboards. Query text, file paths,
and filter values are not exported as labels.
