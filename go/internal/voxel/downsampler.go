// Package voxel provides voxel grid downsampling for point clouds.
// Voxel downsampling reduces point cloud density by keeping only one point
// per voxel cell, which is essential for real-time applications.
package voxel

import (
	"time"

	"github.com/iris3d/go/internal/ply"
)

// Config holds voxel downsampling configuration.
type Config struct {
	// VoxelSize is the size of each voxel cell.
	// Smaller values = more points retained, larger values = more aggressive reduction.
	// Typical values: 0.005 to 0.05 (in scene units)
	VoxelSize float32

	// UseCentroid determines whether to use the centroid of points in a voxel
	// (true) or just keep the first point (false). Centroid is more accurate
	// but slightly slower.
	UseCentroid bool
}

// DefaultConfig returns sensible default configuration.
func DefaultConfig() Config {
	return Config{
		VoxelSize:   0.01,
		UseCentroid: false, // First point is faster
	}
}

// Downsampler performs voxel grid downsampling on point clouds.
type Downsampler struct {
	config Config
}

// New creates a new Downsampler with the given configuration.
func New(config Config) *Downsampler {
	return &Downsampler{config: config}
}

// NewWithSize creates a Downsampler with the specified voxel size.
func NewWithSize(voxelSize float32) *Downsampler {
	return &Downsampler{
		config: Config{
			VoxelSize:   voxelSize,
			UseCentroid: false,
		},
	}
}

// voxelKey is a unique identifier for a voxel cell.
type voxelKey struct {
	x, y, z int64
}

// voxelAccumulator accumulates points in a voxel for centroid computation.
type voxelAccumulator struct {
	sumX, sumY, sumZ float64
	sumR, sumG, sumB float64
	count            int
}

// Downsample reduces the point cloud density using voxel grid filtering.
// Returns the downsampled points and processing time in milliseconds.
func (d *Downsampler) Downsample(points []ply.Point) ([]ply.Point, float64) {
	if len(points) == 0 {
		return points, 0
	}

	start := time.Now()

	if d.config.UseCentroid {
		result := d.downsampleCentroid(points)
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		return result, elapsed
	}

	result := d.downsampleFirst(points)
	elapsed := float64(time.Since(start).Microseconds()) / 1000.0
	return result, elapsed
}

// downsampleFirst keeps the first point encountered in each voxel.
// This is faster than centroid but may be less accurate.
func (d *Downsampler) downsampleFirst(points []ply.Point) []ply.Point {
	voxelSize := float64(d.config.VoxelSize)
	invVoxelSize := 1.0 / voxelSize

	// Map to track which voxels have been seen
	seen := make(map[voxelKey]struct{}, len(points)/10)

	// Pre-allocate result with estimated capacity
	result := make([]ply.Point, 0, len(points)/10)

	for i := range points {
		p := &points[i]

		// Compute voxel indices
		key := voxelKey{
			x: int64(float64(p.X) * invVoxelSize),
			y: int64(float64(p.Y) * invVoxelSize),
			z: int64(float64(p.Z) * invVoxelSize),
		}

		// Only keep point if voxel hasn't been seen
		if _, exists := seen[key]; !exists {
			seen[key] = struct{}{}
			result = append(result, *p)
		}
	}

	return result
}

// downsampleCentroid computes the centroid of all points in each voxel.
// This produces more accurate results but is slower.
func (d *Downsampler) downsampleCentroid(points []ply.Point) []ply.Point {
	voxelSize := float64(d.config.VoxelSize)
	invVoxelSize := 1.0 / voxelSize

	// Map to accumulate points per voxel
	voxels := make(map[voxelKey]*voxelAccumulator, len(points)/10)

	// Accumulate points
	for i := range points {
		p := &points[i]

		key := voxelKey{
			x: int64(float64(p.X) * invVoxelSize),
			y: int64(float64(p.Y) * invVoxelSize),
			z: int64(float64(p.Z) * invVoxelSize),
		}

		acc, exists := voxels[key]
		if !exists {
			acc = &voxelAccumulator{}
			voxels[key] = acc
		}

		acc.sumX += float64(p.X)
		acc.sumY += float64(p.Y)
		acc.sumZ += float64(p.Z)
		acc.sumR += float64(p.R)
		acc.sumG += float64(p.G)
		acc.sumB += float64(p.B)
		acc.count++
	}

	// Compute centroids
	result := make([]ply.Point, 0, len(voxels))
	for _, acc := range voxels {
		n := float64(acc.count)
		result = append(result, ply.Point{
			X: float32(acc.sumX / n),
			Y: float32(acc.sumY / n),
			Z: float32(acc.sumZ / n),
			R: uint8(acc.sumR / n),
			G: uint8(acc.sumG / n),
			B: uint8(acc.sumB / n),
		})
	}

	return result
}

// EstimateVoxelSize suggests a voxel size based on point cloud bounds.
// targetReduction is the desired reduction factor (e.g., 0.1 = keep 10% of points).
func EstimateVoxelSize(points []ply.Point, targetReduction float32) float32 {
	if len(points) < 2 {
		return 0.01
	}

	// Find bounding box
	minX, minY, minZ := points[0].X, points[0].Y, points[0].Z
	maxX, maxY, maxZ := points[0].X, points[0].Y, points[0].Z

	for i := 1; i < len(points); i++ {
		p := &points[i]
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
		if p.Z < minZ {
			minZ = p.Z
		}
		if p.Z > maxZ {
			maxZ = p.Z
		}
	}

	// Compute diagonal
	dx := maxX - minX
	dy := maxY - minY
	dz := maxZ - minZ

	// Estimate voxel size based on target reduction and bounding box
	// Assuming uniform distribution, reduction ~ (voxelSize / diagonal)^3
	diagonal := float32(0)
	if dx > diagonal {
		diagonal = dx
	}
	if dy > diagonal {
		diagonal = dy
	}
	if dz > diagonal {
		diagonal = dz
	}

	if diagonal < 0.001 {
		return 0.01
	}

	// Very rough estimate
	voxelSize := diagonal * targetReduction / 10.0

	// Clamp to reasonable range
	if voxelSize < 0.001 {
		voxelSize = 0.001
	}
	if voxelSize > 1.0 {
		voxelSize = 1.0
	}

	return voxelSize
}
