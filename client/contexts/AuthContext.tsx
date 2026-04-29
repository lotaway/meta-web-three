import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { userApi, DEFAULT_USER_ID } from '@/api/generated';
import type { UserDTO } from '@/src/generated/api/models';

const AUTH_TOKEN_KEY = '@meta_web_three:auth_token';
const USER_ID_KEY = '@meta_web_three:user_id';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: UserDTO | null;
  token: string | null;
  userId: number | null;
  login: (token: string, userId: number) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<UserDTO | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [userId, setUserId] = useState<number | null>(null);

  useEffect(() => {
    loadAuthState();
  }, []);

  async function loadAuthState() {
    try {
      const [storedToken, storedUserId] = await Promise.all([
        AsyncStorage.getItem(AUTH_TOKEN_KEY),
        AsyncStorage.getItem(USER_ID_KEY),
      ]);

      if (storedToken && storedUserId) {
        const uid = parseInt(storedUserId, 10);
        setToken(storedToken);
        setUserId(uid);
        setIsAuthenticated(true);
        
        // 获取用户信息
        try {
          const response = await userApi.info({ xUserId: uid });
          if (response.data) {
            setUser(response.data);
          }
        } catch (error) {
          console.error('Failed to fetch user info:', error);
          // Token 可能已过期，清除认证状态
          await clearAuthState();
        }
      }
    } catch (error) {
      console.error('Failed to load auth state:', error);
    } finally {
      setIsLoading(false);
    }
  }

  async function login(newToken: string, newUserId: number) {
    try {
      await Promise.all([
        AsyncStorage.setItem(AUTH_TOKEN_KEY, newToken),
        AsyncStorage.setItem(USER_ID_KEY, newUserId.toString()),
      ]);

      setToken(newToken);
      setUserId(newUserId);
      setIsAuthenticated(true);

      // 获取用户信息
      const response = await userApi.info({ xUserId: newUserId });
      if (response.data) {
        setUser(response.data);
      }
    } catch (error) {
      console.error('Failed to login:', error);
      throw error;
    }
  }

  async function logout() {
    await clearAuthState();
    setUser(null);
    setToken(null);
    setUserId(null);
    setIsAuthenticated(false);
  }

  async function clearAuthState() {
    try {
      await Promise.all([
        AsyncStorage.removeItem(AUTH_TOKEN_KEY),
        AsyncStorage.removeItem(USER_ID_KEY),
      ]);
    } catch (error) {
      console.error('Failed to clear auth state:', error);
    }
  }

  async function refreshUser() {
    if (!userId) return;
    
    try {
      const response = await userApi.info({ xUserId: userId });
      if (response.data) {
        setUser(response.data);
      }
    } catch (error) {
      console.error('Failed to refresh user:', error);
      throw error;
    }
  }

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        isLoading,
        user,
        token,
        userId,
        login,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
