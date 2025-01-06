import { Buffer } from 'buffer'; // latest
import * as z from 'zod'; // ^3.22.0
import * as t from 'io-ts'; // ^2.2.20
import { LIDAR_SCAN_SETTINGS } from '../constants/lidar.constants';

// Global constants
export const MAX_SCAN_FREQUENCY = 30 as const;
export const MIN_RESOLUTION = 0.01 as const;
export const MAX_RANGE = 5.0 as const;
export const MAX_PROCESSING_TIME = 50 as const;
export const SCAN_ID_PREFIX = 'LIDAR_SCAN_' as const;

// Branded type utilities
type Brand<K, T> = K & { readonly _brand: T };
type ScanFrequency = Brand<number, 'ScanFrequency'>;
type ScanResolution = Brand<number, 'ScanResolution'>;
type ScanRange = Brand<number, 'ScanRange'>;
type ScanId = Brand<string, 'ScanId'>;

/**
 * Immutable 3D point coordinate type
 */
export type Point3D = {
    readonly x: number;
    readonly y: number;
    readonly z: number;
};

/**
 * Immutable point cloud data structure with raw buffer support
 */
export type PointCloudData = {
    readonly points: ReadonlyArray<Point3D>;
    readonly timestamp: number;
    readonly rawData: Buffer;
};

/**
 * Scan quality levels enum
 */
export enum ScanQuality {
    HIGH = 'HIGH',
    MEDIUM = 'MEDIUM',
    LOW = 'LOW'
}

/**
 * Type-safe configuration with branded types for LiDAR scan parameters
 */
export type ScanConfig = {
    readonly frequency: ScanFrequency;
    readonly resolution: ScanResolution;
    readonly range: ScanRange;
};

/**
 * Enhanced type-safe metadata with branded scan ID
 */
export type ScanMetadata = {
    readonly scanId: ScanId;
    readonly timestamp: number;
    readonly processingTime: number;
    readonly quality: ScanQuality;
};

/**
 * Immutable scan state with strict null checks
 */
export type ScanState = {
    readonly isActive: boolean;
    readonly currentScan: PointCloudData | null;
    readonly metadata: ScanMetadata | null;
};

// Zod schemas for runtime validation
export const Point3DSchema = z.object({
    x: z.number(),
    y: z.number(),
    z: z.number()
}).readonly();

export const PointCloudDataSchema = z.object({
    points: z.array(Point3DSchema).readonly(),
    timestamp: z.number(),
    rawData: z.instanceof(Buffer)
}).readonly();

export const ScanConfigSchema = z.object({
    frequency: z.number()
        .min(0)
        .max(LIDAR_SCAN_SETTINGS.SCAN_FREQUENCY)
        .brand<'ScanFrequency'>(),
    resolution: z.number()
        .min(LIDAR_SCAN_SETTINGS.RESOLUTION)
        .brand<'ScanResolution'>(),
    range: z.number()
        .max(LIDAR_SCAN_SETTINGS.MAX_RANGE)
        .brand<'ScanRange'>()
}).readonly();

export const ScanMetadataSchema = z.object({
    scanId: z.string()
        .startsWith(SCAN_ID_PREFIX)
        .brand<'ScanId'>(),
    timestamp: z.number(),
    processingTime: z.number().max(MAX_PROCESSING_TIME),
    quality: z.nativeEnum(ScanQuality)
}).readonly();

// io-ts codecs for advanced runtime type validation
export const Point3DCodec = t.readonly(t.type({
    x: t.number,
    y: t.number,
    z: t.number
}));

export const PointCloudDataCodec = t.readonly(t.type({
    points: t.readonlyArray(Point3DCodec),
    timestamp: t.number,
    rawData: t.unknown // Buffer type handled separately
}));

export const ScanQualityCodec = t.keyof({
    [ScanQuality.HIGH]: null,
    [ScanQuality.MEDIUM]: null,
    [ScanQuality.LOW]: null
});

export const ScanStateCodec = t.readonly(t.type({
    isActive: t.boolean,
    currentScan: t.union([PointCloudDataCodec, t.null]),
    metadata: t.union([
        t.readonly(t.type({
            scanId: t.string,
            timestamp: t.number,
            processingTime: t.number,
            quality: ScanQualityCodec
        })),
        t.null
    ])
}));