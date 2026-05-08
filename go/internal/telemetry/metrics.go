// Package telemetry provides Prometheus metrics for Iris3D.
package telemetry

import (
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds all Prometheus metrics for Iris3D.
type Metrics struct {
	// Request metrics
	requestsTotal   *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec

	// Inference metrics
	inferenceDuration *prometheus.HistogramVec
	modelRequests     *prometheus.CounterVec

	// Point cloud metrics
	pointsGenerated prometheus.Counter
	pointsAfterDS   prometheus.Counter

	// Cache metrics
	cacheHits   prometheus.Counter
	cacheMisses prometheus.Counter

	// System metrics
	activeRequests prometheus.Gauge

	mu sync.Mutex
}

// NewMetrics creates and registers all Prometheus metrics.
func NewMetrics(namespace string) *Metrics {
	m := &Metrics{
		requestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "requests_total",
				Help:      "Total number of requests by method and status",
			},
			[]string{"method", "status"},
		),

		requestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "request_duration_seconds",
				Help:      "Request duration in seconds",
				Buckets:   prometheus.ExponentialBuckets(0.01, 2, 10),
			},
			[]string{"method"},
		),

		inferenceDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: namespace,
				Name:      "inference_duration_seconds",
				Help:      "Model inference duration in seconds",
				Buckets:   prometheus.ExponentialBuckets(0.01, 2, 8),
			},
			[]string{"model"},
		),

		modelRequests: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "model_requests_total",
				Help:      "Total requests per depth model",
			},
			[]string{"model"},
		),

		pointsGenerated: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "points_generated_total",
				Help:      "Total number of 3D points generated",
			},
		),

		pointsAfterDS: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "points_after_downsampling_total",
				Help:      "Total points after voxel downsampling",
			},
		),

		cacheHits: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_hits_total",
				Help:      "Total cache hits",
			},
		),

		cacheMisses: promauto.NewCounter(
			prometheus.CounterOpts{
				Namespace: namespace,
				Name:      "cache_misses_total",
				Help:      "Total cache misses",
			},
		),

		activeRequests: promauto.NewGauge(
			prometheus.GaugeOpts{
				Namespace: namespace,
				Name:      "active_requests",
				Help:      "Number of active requests",
			},
		),
	}

	return m
}

// RecordRequest records a request completion.
func (m *Metrics) RecordRequest(method, status string, durationSeconds float64) {
	m.requestsTotal.WithLabelValues(method, status).Inc()
	m.requestDuration.WithLabelValues(method).Observe(durationSeconds)
}

// RecordInference records model inference.
func (m *Metrics) RecordInference(model string, durationMs float64) {
	m.inferenceDuration.WithLabelValues(model).Observe(durationMs / 1000.0)
	m.modelRequests.WithLabelValues(model).Inc()
}

// RecordPointsGenerated records generated point count.
func (m *Metrics) RecordPointsGenerated(count int64) {
	m.pointsGenerated.Add(float64(count))
}

// RecordPointsAfterDownsampling records points after downsampling.
func (m *Metrics) RecordPointsAfterDownsampling(count int64) {
	m.pointsAfterDS.Add(float64(count))
}

// RecordCacheHit records a cache hit.
func (m *Metrics) RecordCacheHit() {
	m.cacheHits.Inc()
}

// RecordCacheMiss records a cache miss.
func (m *Metrics) RecordCacheMiss() {
	m.cacheMisses.Inc()
}

// IncActiveRequests increments active request count.
func (m *Metrics) IncActiveRequests() {
	m.activeRequests.Inc()
}

// DecActiveRequests decrements active request count.
func (m *Metrics) DecActiveRequests() {
	m.activeRequests.Dec()
}

// Handler returns HTTP handler for metrics endpoint.
func (m *Metrics) Handler() http.Handler {
	return promhttp.Handler()
}

// MetricsMiddleware wraps HTTP handler with metrics.
func (m *Metrics) MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.IncActiveRequests()
		defer m.DecActiveRequests()
		next.ServeHTTP(w, r)
	})
}
