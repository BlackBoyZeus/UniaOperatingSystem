import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import styled from '@emotion/styled';
import AuthLayout from '../layouts/AuthLayout';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import { validateUser } from '../utils/validation.utils';
import { UI_CONSTANTS } from '../constants/ui.constants';

// Registration form data interface with hardware validation
interface RegisterFormData {
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
  hardwareToken: string;
  tpmValidation: boolean;
  powerMode: 'high' | 'balanced' | 'power-save';
  hdrEnabled: boolean;
}

// GPU-accelerated form container
const StyledRegisterForm = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  width: 100%;
  max-width: 400px;
  will-change: transform, opacity;
  transform: translateZ(0);
  backface-visibility: hidden;
  perspective: 1000px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
`;

// HDR-aware title component
const StyledTitle = styled.h1`
  font-size: 2rem;
  font-family: var(--font-family-gaming);
  text-align: center;
  color: ${({ theme }) => theme.hdrEnabled ? 
    'color(display-p3 0.486 0.302 1)' : 
    theme.colors.primary};
  margin-bottom: 2rem;
  text-shadow: ${({ theme }) => theme.hdrEnabled ?
    '0 0 10px color(display-p3 0.6 0.4 1)' :
    'none'};
`;

// Error message with GPU-accelerated animations
const StyledError = styled.div`
  color: ${({ theme }) => theme.hdrEnabled ?
    'color(display-p3 1 0 0)' :
    theme.colors.error};
  font-size: 0.875rem;
  text-align: center;
  animation: shake 0.5s cubic-bezier(0.36, 0, 0.66, -0.56);
  transform: translateZ(0);
  backface-visibility: hidden;

  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
  }
`;

const Register: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState<RegisterFormData>({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    hardwareToken: '',
    tpmValidation: false,
    powerMode: 'balanced',
    hdrEnabled: window.matchMedia('(dynamic-range: high)').matches
  });
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Power-aware form updates
  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    requestAnimationFrame(() => {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
      setError(null);
    });
  }, []);

  // Hardware token validation with TPM
  const validateHardwareToken = useCallback(async (token: string): Promise<boolean> => {
    try {
      const tpm = await window.navigator.credentials.get({
        publicKey: {
          challenge: new Uint8Array(32),
          rpId: window.location.hostname,
          allowCredentials: [],
          userVerification: 'required'
        }
      });
      return !!tpm;
    } catch (err) {
      console.error('TPM validation failed:', err);
      return false;
    }
  }, []);

  // Enhanced form submission with security checks
  const handleSubmit = useCallback(async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (isSubmitting) return;
    setIsSubmitting(true);
    setError(null);

    try {
      // Validate form data
      if (!formData.username || !formData.email || !formData.password || !formData.confirmPassword) {
        throw new Error('All fields are required');
      }

      if (formData.password !== formData.confirmPassword) {
        throw new Error('Passwords do not match');
      }

      // Validate user data
      await validateUser({
        username: formData.username,
        email: formData.email
      });

      // Hardware token validation
      const isValidToken = await validateHardwareToken(formData.hardwareToken);
      if (!isValidToken) {
        throw new Error('Hardware token validation failed');
      }

      // Submit registration
      const response = await fetch('/api/v1/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Hardware-Token': formData.hardwareToken
        },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password,
          tpmValidation: formData.tpmValidation
        })
      });

      if (!response.ok) {
        throw new Error('Registration failed');
      }

      // Navigate to login on success
      navigate('/login', { 
        state: { 
          message: 'Registration successful! Please log in.' 
        } 
      });

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setIsSubmitting(false);
    }
  }, [formData, isSubmitting, navigate, validateHardwareToken]);

  // HDR support detection
  useEffect(() => {
    const hdrQuery = window.matchMedia('(dynamic-range: high)');
    const handleHDRChange = (e: MediaQueryListEvent) => {
      setFormData(prev => ({
        ...prev,
        hdrEnabled: e.matches
      }));
    };
    
    hdrQuery.addEventListener('change', handleHDRChange);
    return () => hdrQuery.removeEventListener('change', handleHDRChange);
  }, []);

  return (
    <AuthLayout>
      <StyledTitle>Register</StyledTitle>
      <StyledRegisterForm onSubmit={handleSubmit} noValidate>
        <Input
          id="username"
          name="username"
          type="text"
          value={formData.username}
          onChange={handleInputChange}
          placeholder="Username"
          required
          autoFocus
          gpuAccelerated
          performanceMode={formData.powerMode === 'high'}
        />
        <Input
          id="email"
          name="email"
          type="email"
          value={formData.email}
          onChange={handleInputChange}
          placeholder="Email"
          required
          gpuAccelerated
          performanceMode={formData.powerMode === 'high'}
        />
        <Input
          id="password"
          name="password"
          type="password"
          value={formData.password}
          onChange={handleInputChange}
          placeholder="Password"
          required
          gpuAccelerated
          performanceMode={formData.powerMode === 'high'}
        />
        <Input
          id="confirmPassword"
          name="confirmPassword"
          type="password"
          value={formData.confirmPassword}
          onChange={handleInputChange}
          placeholder="Confirm Password"
          required
          gpuAccelerated
          performanceMode={formData.powerMode === 'high'}
        />
        <Button
          type="submit"
          disabled={isSubmitting}
          fullWidth
          variant="primary"
          size="large"
          enableHaptic
          hdrMode={formData.hdrEnabled ? 'enabled' : 'disabled'}
          powerSaveAware
        >
          {isSubmitting ? 'Registering...' : 'Register'}
        </Button>
        {error && <StyledError role="alert">{error}</StyledError>}
      </StyledRegisterForm>
    </AuthLayout>
  );
};

export default Register;