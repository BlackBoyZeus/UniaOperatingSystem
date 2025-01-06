import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from '@emotion/styled';
import debounce from 'lodash/debounce';

import Button from '../components/common/Button';
import Input from '../components/common/Input';
import { useAuth } from '../hooks/useAuth';

// Interfaces
interface LoginFormState {
  username: string;
  password: string;
  hardwareToken: string;
  trustScore: number;
  lastAttempt: Date | null;
}

interface SecurityMetrics {
  failedAttempts: number;
  lastFailure: Date | null;
  deviceTrust: number;
  tokenValidation: boolean;
}

// Styled Components with HDR and power-aware features
const LoginContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: calc(var(--spacing-unit) * 2);
  background: var(--color-background);
  color-scheme: dark light;
  isolation: isolate;
`;

const LoginForm = styled.form`
  width: 100%;
  max-width: 420px;
  padding: calc(var(--spacing-unit) * 3);
  background: color(display-p3 var(--color-surface));
  border: 1px solid var(--color-primary);
  border-radius: calc(var(--spacing-unit));
  box-shadow: var(--effect-glow);
  transform: var(--animation-gpu);
  will-change: transform, opacity;
  transition: transform var(--animation-duration) cubic-bezier(0.4, 0, 0.2, 1);

  @media (dynamic-range: high) {
    box-shadow: var(--effect-glow);
    border-color: var(--color-primary-hdr);
  }

  @media (prefers-reduced-motion: reduce) {
    transition: none;
    will-change: auto;
  }
`;

const FormGroup = styled.div`
  margin-bottom: calc(var(--spacing-unit) * 2);
`;

const ErrorMessage = styled.div`
  color: color(display-p3 1 0.3 0.3);
  margin-top: calc(var(--spacing-unit));
  font-size: 0.875rem;
  transform: var(--animation-gpu);
  animation: errorShake 0.6s cubic-bezier(0.36, 0.07, 0.19, 0.97) both;

  @keyframes errorShake {
    10%, 90% { transform: var(--animation-gpu) translate3d(-1px, 0, 0); }
    20%, 80% { transform: var(--animation-gpu) translate3d(2px, 0, 0); }
    30%, 50%, 70% { transform: var(--animation-gpu) translate3d(-4px, 0, 0); }
    40%, 60% { transform: var(--animation-gpu) translate3d(4px, 0, 0); }
  }
`;

const Login: React.FC = () => {
  const navigate = useNavigate();
  const { login, isLoading, error: authError } = useAuth();
  const formRef = useRef<HTMLFormElement>(null);

  // State management
  const [formState, setFormState] = useState<LoginFormState>({
    username: '',
    password: '',
    hardwareToken: '',
    trustScore: 100,
    lastAttempt: null
  });

  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics>({
    failedAttempts: 0,
    lastFailure: null,
    deviceTrust: 100,
    tokenValidation: false
  });

  const [formError, setFormError] = useState<string | null>(null);

  // Hardware token detection
  useEffect(() => {
    const detectHardwareToken = async () => {
      try {
        const token = await navigator.credentials?.get({
          publicKey: {
            challenge: new Uint8Array(32),
            rpId: window.location.hostname,
            timeout: 60000
          }
        });
        
        if (token) {
          setFormState(prev => ({
            ...prev,
            hardwareToken: (token as any).id
          }));
          setSecurityMetrics(prev => ({
            ...prev,
            tokenValidation: true
          }));
        }
      } catch (err) {
        setSecurityMetrics(prev => ({
          ...prev,
          deviceTrust: prev.deviceTrust * 0.8
        }));
      }
    };

    detectHardwareToken();
  }, []);

  // Debounced input handler with security validation
  const handleInputChange = useCallback(
    debounce((event: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value } = event.target;
      
      // Real-time input validation
      const sanitizedValue = value.replace(/[<>]/g, '');
      
      setFormState(prev => ({
        ...prev,
        [name]: sanitizedValue
      }));

      // Clear relevant errors
      setFormError(null);
    }, 16),
    []
  );

  // Enhanced form submission with security checks
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();

    try {
      // Security pre-checks
      if (securityMetrics.failedAttempts >= 5) {
        throw new Error('Maximum login attempts exceeded. Please try again later.');
      }

      if (securityMetrics.deviceTrust < 50) {
        throw new Error('Device trust score too low. Please verify your device.');
      }

      // Attempt login
      await login(
        formState.username,
        formState.password,
        formState.hardwareToken
      );

      // Successful login
      navigate('/dashboard');

    } catch (err) {
      // Update security metrics
      setSecurityMetrics(prev => ({
        ...prev,
        failedAttempts: prev.failedAttempts + 1,
        lastFailure: new Date(),
        deviceTrust: prev.deviceTrust * 0.9
      }));

      setFormError(err instanceof Error ? err.message : 'Authentication failed');
    }
  };

  return (
    <LoginContainer>
      <LoginForm
        ref={formRef}
        onSubmit={handleSubmit}
        className="gaming-theme"
      >
        <FormGroup>
          <Input
            id="username"
            name="username"
            type="text"
            value={formState.username}
            onChange={handleInputChange}
            placeholder="Username"
            required
            disabled={isLoading}
            validation
            securityLevel="high"
            gpuAccelerated
          />
        </FormGroup>

        <FormGroup>
          <Input
            id="password"
            name="password"
            type="password"
            value={formState.password}
            onChange={handleInputChange}
            placeholder="Password"
            required
            disabled={isLoading}
            validation
            securityLevel="high"
            gpuAccelerated
          />
        </FormGroup>

        <Button
          type="submit"
          disabled={isLoading || securityMetrics.failedAttempts >= 5}
          fullWidth
          variant="primary"
          size="large"
          enableHaptic
          hdrMode="auto"
          powerSaveAware
        >
          {isLoading ? 'Authenticating...' : 'Login'}
        </Button>

        {(formError || authError) && (
          <ErrorMessage role="alert">
            {formError || authError}
          </ErrorMessage>
        )}
      </LoginForm>
    </LoginContainer>
  );
};

export default Login;