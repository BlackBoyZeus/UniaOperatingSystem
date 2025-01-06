import React, { useCallback, useRef, useState, useEffect } from 'react';
import classNames from 'classnames'; // ^2.3.2
import { validateUser } from '../../utils/validation.utils'; // Internal import

/**
 * Enhanced props interface for Input component with gaming and security features
 */
interface InputProps {
    id: string;
    name: string;
    type: string;
    value: string;
    placeholder?: string;
    disabled?: boolean;
    required?: boolean;
    className?: string;
    onChange?: (value: string) => void;
    onBlur?: (event: React.FocusEvent<HTMLInputElement>) => void;
    onFocus?: (event: React.FocusEvent<HTMLInputElement>) => void;
    autoFocus?: boolean;
    error?: string;
    validation?: boolean;
    validationType?: 'user' | 'fleet' | 'game';
    securityLevel?: string;
    performanceMode?: boolean;
    gpuAccelerated?: boolean;
}

/**
 * High-performance, secure, and accessible input component optimized for gaming interactions
 */
const Input: React.FC<InputProps> = ({
    id,
    name,
    type,
    value,
    placeholder,
    disabled = false,
    required = false,
    className,
    onChange,
    onBlur,
    onFocus,
    autoFocus = false,
    error,
    validation = false,
    validationType = 'user',
    securityLevel = 'medium',
    performanceMode = true,
    gpuAccelerated = true,
}) => {
    // State management with performance optimizations
    const [internalValue, setInternalValue] = useState(value);
    const [isFocused, setIsFocused] = useState(false);
    const [internalError, setInternalError] = useState(error);
    const inputRef = useRef<HTMLInputElement>(null);
    const frameRef = useRef<number>();

    // Cleanup animation frame on unmount
    useEffect(() => {
        return () => {
            if (frameRef.current) {
                cancelAnimationFrame(frameRef.current);
            }
        };
    }, []);

    /**
     * Enhanced input change handler with validation and security checks
     */
    const handleChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        event.preventDefault();

        // Sanitize input value
        const sanitizedValue = event.target.value.replace(/[<>]/g, '');

        if (performanceMode) {
            // Use requestAnimationFrame for smooth updates
            frameRef.current = requestAnimationFrame(() => {
                setInternalValue(sanitizedValue);
                
                // Validate input if enabled
                if (validation) {
                    try {
                        validateUser({ [name]: sanitizedValue });
                        setInternalError(undefined);
                    } catch (err) {
                        setInternalError((err as Error).message);
                    }
                }

                // Trigger onChange callback
                onChange?.(sanitizedValue);
            });
        } else {
            setInternalValue(sanitizedValue);
            onChange?.(sanitizedValue);
        }
    }, [name, onChange, performanceMode, validation]);

    /**
     * Enhanced blur handler with validation and gaming focus management
     */
    const handleBlur = useCallback((event: React.FocusEvent<HTMLInputElement>) => {
        setIsFocused(false);

        // Run validation on blur if enabled
        if (validation) {
            try {
                validateUser({ [name]: internalValue });
                setInternalError(undefined);
            } catch (err) {
                setInternalError((err as Error).message);
            }
        }

        onBlur?.(event);
    }, [internalValue, name, onBlur, validation]);

    /**
     * Focus handler with gaming-specific optimizations
     */
    const handleFocus = useCallback((event: React.FocusEvent<HTMLInputElement>) => {
        setIsFocused(true);
        onFocus?.(event);
    }, [onFocus]);

    // Compute dynamic class names
    const inputClasses = classNames(
        'input',
        {
            'input--error': internalError,
            'input--disabled': disabled,
            'input--focused': isFocused,
            'input--gpu-accelerated': gpuAccelerated,
            'input--gaming-mode': performanceMode,
            'input--high-performance': performanceMode,
        },
        className
    );

    return (
        <div className="input-wrapper">
            <input
                ref={inputRef}
                id={id}
                name={name}
                type={type}
                value={internalValue}
                placeholder={placeholder}
                disabled={disabled}
                required={required}
                className={inputClasses}
                onChange={handleChange}
                onBlur={handleBlur}
                onFocus={handleFocus}
                autoFocus={autoFocus}
                autoComplete="off" // Disable browser autocomplete for gaming inputs
                data-security-level={securityLevel}
                style={{
                    // GPU acceleration optimizations
                    transform: gpuAccelerated ? 'translate3d(0, 0, 0)' : undefined,
                    willChange: gpuAccelerated ? 'transform, opacity' : undefined,
                    backfaceVisibility: gpuAccelerated ? 'hidden' : undefined,
                }}
                aria-invalid={!!internalError}
                aria-required={required}
            />
            {internalError && (
                <div 
                    className="input__error"
                    role="alert"
                    aria-live="polite"
                >
                    {internalError}
                </div>
            )}
        </div>
    );
};

export default Input;