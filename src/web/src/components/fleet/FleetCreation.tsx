import React, { useState, useCallback, useRef, useEffect } from 'react'; // @version 18.2.0
import { z } from 'zod'; // @version ^3.22.0
import * as Automerge from 'automerge'; // @version ^2.0.0

import { IFleet } from '../../interfaces/fleet.interface';
import useFleet from '../../hooks/useFleet';

// Validation schema for fleet creation form
const fleetFormSchema = z.object({
    name: z.string()
        .min(3, 'Fleet name must be at least 3 characters')
        .max(32, 'Fleet name cannot exceed 32 characters'),
    maxDevices: z.number()
        .min(1, 'Fleet must allow at least 1 device')
        .max(32, 'Fleet cannot exceed 32 devices'),
    isPublic: z.boolean(),
    networkQuality: z.object({
        minLatency: z.number().max(50, 'Maximum latency threshold exceeded'),
        minBandwidth: z.number().min(1000, 'Minimum bandwidth not met')
    }),
    leaderConfig: z.object({
        electionTimeout: z.number().min(1000).max(10000),
        heartbeatInterval: z.number().min(500).max(5000)
    })
});

// Props interface with enhanced type safety
interface FleetCreationProps {
    onSuccess: (fleet: IFleet) => void;
    onCancel: () => void;
    className?: string;
    networkRequirements?: {
        minLatency: number;
        minBandwidth: number;
    };
    hardwareCapabilities?: {
        lidarSupported: boolean;
        meshNetworkSupported: boolean;
        maxFleetSize: number;
    };
}

// Form data interface with network and hardware requirements
interface FleetFormData {
    name: string;
    maxDevices: number;
    isPublic: boolean;
    networkQuality: {
        minLatency: number;
        minBandwidth: number;
    };
    leaderConfig: {
        electionTimeout: number;
        heartbeatInterval: number;
    };
}

/**
 * Enhanced FleetCreation component with real-time validation and P2P setup
 */
export const FleetCreation: React.FC<FleetCreationProps> = ({
    onSuccess,
    onCancel,
    className = '',
    networkRequirements,
    hardwareCapabilities
}) => {
    // Fleet management hook
    const { createFleet, joinFleet, initializeP2P, setupLeaderElection } = useFleet();

    // Form state with validation
    const [formData, setFormData] = useState<FleetFormData>({
        name: '',
        maxDevices: 32,
        isPublic: true,
        networkQuality: {
            minLatency: networkRequirements?.minLatency || 50,
            minBandwidth: networkRequirements?.minBandwidth || 1000
        },
        leaderConfig: {
            electionTimeout: 5000,
            heartbeatInterval: 1000
        }
    });

    // Error and validation state
    const [errors, setErrors] = useState<Record<string, string>>({});
    const [isSubmitting, setIsSubmitting] = useState(false);

    // WebWorker for validation
    const validationWorker = useRef<Worker | null>(null);

    // Initialize validation worker
    useEffect(() => {
        validationWorker.current = new Worker(
            new URL('../../workers/fleetValidation.worker.ts', import.meta.url)
        );

        validationWorker.current.onmessage = (event) => {
            const { isValid, errors } = event.data;
            if (!isValid) {
                setErrors(errors);
            }
        };

        return () => {
            validationWorker.current?.terminate();
        };
    }, []);

    // Debounced form validation
    const validateForm = useCallback(async (data: FleetFormData): Promise<boolean> => {
        try {
            const validated = await fleetFormSchema.parseAsync(data);
            setErrors({});
            return true;
        } catch (error) {
            if (error instanceof z.ZodError) {
                const formattedErrors: Record<string, string> = {};
                error.errors.forEach((err) => {
                    if (err.path) {
                        formattedErrors[err.path.join('.')] = err.message;
                    }
                });
                setErrors(formattedErrors);
            }
            return false;
        }
    }, []);

    // Handle form field changes with real-time validation
    const handleChange = useCallback((
        e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
    ) => {
        const { name, value, type } = e.target;
        const newValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked :
                        type === 'number' ? Number(value) : value;

        setFormData(prev => {
            const updated = { ...prev };
            if (name.includes('.')) {
                const [parent, child] = name.split('.');
                (updated as any)[parent] = {
                    ...(updated as any)[parent],
                    [child]: newValue
                };
            } else {
                (updated as any)[name] = newValue;
            }
            return updated;
        });

        // Trigger validation in WebWorker
        validationWorker.current?.postMessage({
            type: 'validate',
            data: formData
        });
    }, [formData]);

    // Enhanced form submission with P2P setup and leader election
    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSubmitting(true);

        try {
            // Validate form data
            const isValid = await validateForm(formData);
            if (!isValid) {
                return;
            }

            // Verify hardware capabilities
            if (hardwareCapabilities) {
                if (!hardwareCapabilities.meshNetworkSupported) {
                    throw new Error('Mesh networking not supported on this device');
                }
                if (formData.maxDevices > hardwareCapabilities.maxFleetSize) {
                    throw new Error('Fleet size exceeds hardware capabilities');
                }
            }

            // Initialize CRDT state
            const initialState = Automerge.init<IFleet>();
            const fleetDoc = Automerge.change(initialState, 'Initialize fleet', doc => {
                doc.name = formData.name;
                doc.maxDevices = formData.maxDevices;
                doc.members = [];
                doc.networkQuality = formData.networkQuality;
            });

            // Create fleet
            const fleet = await createFleet(formData.name, formData.maxDevices);

            // Initialize P2P connections
            await initializeP2P(fleet.id);

            // Setup leader election if we're the first member
            if (fleet.members.length === 0) {
                await setupLeaderElection({
                    timeout: formData.leaderConfig.electionTimeout,
                    heartbeat: formData.leaderConfig.heartbeatInterval
                });
            }

            // Join fleet as initial member
            await joinFleet(fleet.id);

            onSuccess(fleet);

        } catch (error) {
            console.error('Fleet creation failed:', error);
            setErrors({
                submit: error instanceof Error ? error.message : 'Fleet creation failed'
            });
        } finally {
            setIsSubmitting(false);
        }
    }, [formData, createFleet, joinFleet, initializeP2P, setupLeaderElection, onSuccess, hardwareCapabilities]);

    return (
        <form onSubmit={handleSubmit} className={`fleet-creation ${className}`}>
            <div className="form-group">
                <label htmlFor="name">Fleet Name</label>
                <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={errors.name ? 'error' : ''}
                    disabled={isSubmitting}
                />
                {errors.name && <span className="error-message">{errors.name}</span>}
            </div>

            <div className="form-group">
                <label htmlFor="maxDevices">Maximum Devices</label>
                <input
                    type="number"
                    id="maxDevices"
                    name="maxDevices"
                    value={formData.maxDevices}
                    onChange={handleChange}
                    min={1}
                    max={hardwareCapabilities?.maxFleetSize || 32}
                    className={errors.maxDevices ? 'error' : ''}
                    disabled={isSubmitting}
                />
                {errors.maxDevices && <span className="error-message">{errors.maxDevices}</span>}
            </div>

            <div className="form-group">
                <label htmlFor="isPublic">Public Fleet</label>
                <input
                    type="checkbox"
                    id="isPublic"
                    name="isPublic"
                    checked={formData.isPublic}
                    onChange={handleChange}
                    disabled={isSubmitting}
                />
            </div>

            <div className="form-group">
                <label htmlFor="networkQuality.minLatency">Minimum Latency (ms)</label>
                <input
                    type="number"
                    id="networkQuality.minLatency"
                    name="networkQuality.minLatency"
                    value={formData.networkQuality.minLatency}
                    onChange={handleChange}
                    min={0}
                    max={50}
                    className={errors['networkQuality.minLatency'] ? 'error' : ''}
                    disabled={isSubmitting}
                />
                {errors['networkQuality.minLatency'] && (
                    <span className="error-message">{errors['networkQuality.minLatency']}</span>
                )}
            </div>

            <div className="form-group">
                <label htmlFor="networkQuality.minBandwidth">Minimum Bandwidth (Kbps)</label>
                <input
                    type="number"
                    id="networkQuality.minBandwidth"
                    name="networkQuality.minBandwidth"
                    value={formData.networkQuality.minBandwidth}
                    onChange={handleChange}
                    min={1000}
                    className={errors['networkQuality.minBandwidth'] ? 'error' : ''}
                    disabled={isSubmitting}
                />
                {errors['networkQuality.minBandwidth'] && (
                    <span className="error-message">{errors['networkQuality.minBandwidth']}</span>
                )}
            </div>

            <div className="form-actions">
                <button
                    type="submit"
                    disabled={isSubmitting || Object.keys(errors).length > 0}
                    className="submit-button"
                >
                    {isSubmitting ? 'Creating Fleet...' : 'Create Fleet'}
                </button>
                <button
                    type="button"
                    onClick={onCancel}
                    disabled={isSubmitting}
                    className="cancel-button"
                >
                    Cancel
                </button>
            </div>

            {errors.submit && (
                <div className="submit-error">
                    {errors.submit}
                </div>
            )}
        </form>
    );
};

export default FleetCreation;