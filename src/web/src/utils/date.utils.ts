import { format, differenceInMilliseconds, addMilliseconds } from 'date-fns'; // v2.30.0

// Cache for memoized timestamp formatting
const timestampCache = new Map<number, string>();

/**
 * Formats game session duration in HH:mm:ss format with timezone awareness
 * @param startTime - Session start timestamp
 * @returns Formatted duration string with proper timezone handling
 * @throws Error if startTime is invalid
 */
export const formatGameTime = (startTime: Date): string => {
  try {
    if (!(startTime instanceof Date) || isNaN(startTime.getTime())) {
      throw new Error('Invalid start time provided');
    }

    const durationMs = performance.now() - startTime.getTime();
    if (durationMs < 0) {
      throw new Error('Start time cannot be in the future');
    }

    const hours = Math.floor(durationMs / 3600000);
    const minutes = Math.floor((durationMs % 3600000) / 60000);
    const seconds = Math.floor((durationMs % 60000) / 1000);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  } catch (error) {
    console.error('Error formatting game time:', error);
    return '--:--:--';
  }
};

/**
 * Formats performance metric timestamps with millisecond precision
 * @param timestamp - Performance metric timestamp
 * @returns High-precision formatted timestamp
 * @throws Error if timestamp is invalid
 */
export const formatMetricTimestamp = (timestamp: Date): string => {
  try {
    if (!(timestamp instanceof Date) || isNaN(timestamp.getTime())) {
      throw new Error('Invalid timestamp provided');
    }

    const cacheKey = timestamp.getTime();
    const cached = timestampCache.get(cacheKey);
    if (cached) {
      return cached;
    }

    const formatted = format(timestamp, 'HH:mm:ss.SSS');
    timestampCache.set(cacheKey, formatted);

    // Limit cache size to prevent memory leaks
    if (timestampCache.size > 1000) {
      const oldestKey = timestampCache.keys().next().value;
      timestampCache.delete(oldestKey);
    }

    return formatted;
  } catch (error) {
    console.error('Error formatting metric timestamp:', error);
    return '--:--:--.---';
  }
};

/**
 * Calculates precise duration between timestamps
 * @param startTime - Start timestamp
 * @param endTime - End timestamp
 * @returns Duration in milliseconds
 * @throws Error if timestamps are invalid
 */
export const calculateDuration = (startTime: Date, endTime: Date): number => {
  try {
    if (!(startTime instanceof Date) || !(endTime instanceof Date)) {
      throw new Error('Invalid timestamp(s) provided');
    }

    if (isNaN(startTime.getTime()) || isNaN(endTime.getTime())) {
      throw new Error('Invalid date values');
    }

    const duration = differenceInMilliseconds(endTime, startTime);
    if (duration < 0) {
      throw new Error('End time cannot be before start time');
    }

    return duration;
  } catch (error) {
    console.error('Error calculating duration:', error);
    return 0;
  }
};

/**
 * Formats fleet uptime with dynamic unit selection
 * @param startTime - Fleet start timestamp
 * @returns Localized uptime string
 * @throws Error if startTime is invalid
 */
export const formatFleetUptime = (startTime: Date): string => {
  try {
    if (!(startTime instanceof Date) || isNaN(startTime.getTime())) {
      throw new Error('Invalid start time provided');
    }

    const uptimeMs = Date.now() - startTime.getTime();
    if (uptimeMs < 0) {
      throw new Error('Start time cannot be in the future');
    }

    const hours = Math.floor(uptimeMs / 3600000);
    const minutes = Math.floor((uptimeMs % 3600000) / 60000);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m`;
    } else {
      return 'Just started';
    }
  } catch (error) {
    console.error('Error formatting fleet uptime:', error);
    return 'Unknown';
  }
};

/**
 * Adds duration to timestamp with validation
 * @param timestamp - Base timestamp
 * @param durationMs - Duration to add in milliseconds
 * @returns New date object with added duration
 * @throws Error if inputs are invalid
 */
export const addDuration = (timestamp: Date, durationMs: number): Date => {
  try {
    if (!(timestamp instanceof Date) || isNaN(timestamp.getTime())) {
      throw new Error('Invalid timestamp provided');
    }

    if (typeof durationMs !== 'number' || isNaN(durationMs)) {
      throw new Error('Invalid duration provided');
    }

    if (durationMs < 0) {
      throw new Error('Duration cannot be negative');
    }

    // Use date-fns for precise addition with timezone handling
    return addMilliseconds(timestamp, durationMs);
  } catch (error) {
    console.error('Error adding duration:', error);
    return new Date();
  }
};