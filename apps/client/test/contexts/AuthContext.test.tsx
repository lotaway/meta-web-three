import { act, render, waitFor } from '@testing-library/react-native';
import React from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ssoApi, userApi } from '@/api/generated';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';

jest.mock('@/api/generated', () => ({
  ssoApi: {
    loginByPhone: jest.fn(),
    getAuthCode: jest.fn(),
    updatePassword: jest.fn(),
    info1: jest.fn(),
    login: jest.fn(),
  },
  userApi: {
    info: jest.fn(),
    signIn: jest.fn(),
  },
  DEFAULT_USER_ID: 0,
}));

let auth: ReturnType<typeof useAuth>;

function Probe() {
  auth = useAuth();
  return null;
}

function renderAuth() {
  const view = render(
    <AuthProvider>
      <Probe />
    </AuthProvider>,
  );
  return view;
}

const TOO_MANY_REQUESTS = Object.assign(new Error('Too many requests'), { code: '429' });

describe('AuthContext auth flows', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    (userApi.info as jest.Mock).mockResolvedValue({ data: null });
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('logs in with phone and persists the session', async () => {
    (ssoApi.loginByPhone as jest.Mock).mockResolvedValue({ data: { token: 'tok', tokenHead: 'head' } });
    (ssoApi.info1 as jest.Mock).mockResolvedValue({ data: { id: 7, username: 'user' } });
    renderAuth();
    await waitFor(() => expect(auth.isLoading).toBe(false));

    await act(async () => {
      await auth.loginWithPhone('13800000000', '123456');
    });

    expect(ssoApi.loginByPhone).toHaveBeenCalledWith({ telephone: '13800000000', authCode: '123456' });
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('@meta_web_three:auth_token', 'headtok');
    expect(auth.isAuthenticated).toBe(true);
    expect(auth.userId).toBe(7);
  });

  it('rejects an invalid phone login response', async () => {
    (ssoApi.loginByPhone as jest.Mock).mockResolvedValue({ data: null });
    renderAuth();
    await waitFor(() => expect(auth.isLoading).toBe(false));

    await expect(auth.loginWithPhone('13800000000', '123456')).rejects.toThrow('Invalid phone login response');
    expect(auth.isAuthenticated).toBe(false);
  });

  it('requests an auth code and surfaces rate limiting', async () => {
    (ssoApi.getAuthCode as jest.Mock).mockResolvedValueOnce({ data: 'ok' });
    renderAuth();
    await waitFor(() => expect(auth.isLoading).toBe(false));

    await act(async () => {
      await auth.getAuthCode('13800000000');
    });
    expect(ssoApi.getAuthCode).toHaveBeenCalledWith({ telephone: '13800000000' });

    (ssoApi.getAuthCode as jest.Mock).mockRejectedValueOnce(TOO_MANY_REQUESTS);
    await expect(auth.getAuthCode('13800000000')).rejects.toThrow('Too many requests');
  });

  it('resets the password and propagates failures', async () => {
    (ssoApi.updatePassword as jest.Mock).mockResolvedValueOnce({ data: null });
    renderAuth();
    await waitFor(() => expect(auth.isLoading).toBe(false));

    await act(async () => {
      await auth.forgotPassword('13800000000', 'newPass', '123456');
    });
    expect(ssoApi.updatePassword).toHaveBeenCalledWith({
      telephone: '13800000000',
      password: 'newPass',
      authCode: '123456',
    });

    (ssoApi.updatePassword as jest.Mock).mockRejectedValueOnce(TOO_MANY_REQUESTS);
    await expect(auth.forgotPassword('13800000000', 'newPass', '123456')).rejects.toThrow('Too many requests');
  });
});
