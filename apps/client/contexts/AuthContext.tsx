import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { userApi, ssoApi, DEFAULT_USER_ID } from '@/api/generated';
import type { UserDTO, LoginResponseDTO } from '@/src/generated/api/models';

const AUTH_TOKEN_KEY = '@meta_web_three:auth_token';
const USER_ID_KEY = '@meta_web_three:user_id';
const USER_DATA_KEY = '@meta_web_three:user_data';

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: UserDTO | null;
  token: string | null;
  userId: number | null;
  login: (token: string, userId: number, userData?: UserDTO) => Promise<void>;
  loginWithCredentials: (email: string, password: string) => Promise<LoginResponseDTO>;
  loginWithSso: (username: string, password: string) => Promise<{ token: string; tokenHead: string }>;
  loginWithPhone: (phone: string, authCode: string) => Promise<void>;
  forgotPassword: (telephone: string, password: string, authCode: string) => Promise<void>;
  getAuthCode: (telephone: string) => Promise<void>;
  register: (email: string, password: string, referrerId?: number) => Promise<void>;
  registerWithSso: (username: string, password: string) => Promise<void>;
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
      const [storedToken, storedUserId, storedUserData] = await Promise.all([
        AsyncStorage.getItem(AUTH_TOKEN_KEY),
        AsyncStorage.getItem(USER_ID_KEY),
        AsyncStorage.getItem(USER_DATA_KEY),
      ]);

      if (storedToken && storedUserId) {
        const uid = parseInt(storedUserId, 10);
        setToken(storedToken);
        setUserId(uid);
        setIsAuthenticated(true);
        
        if (storedUserData) {
          try {
            setUser(JSON.parse(storedUserData));
          } catch {
            // userData parse error, will refresh from API
          }
        }
        
        // 获取最新用户信息
        try {
          const response = await userApi.info({ xUserId: uid });
          if (response.data) {
            setUser(response.data);
            await AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(response.data));
          }
        } catch (error) {
          console.error('Failed to fetch user info:', error);
          await clearAuthState();
        }
      }
    } catch (error) {
      console.error('Failed to load auth state:', error);
    } finally {
      setIsLoading(false);
    }
  }

  async function login(newToken: string, newUserId: number, userData?: UserDTO) {
    try {
      await Promise.all([
        AsyncStorage.setItem(AUTH_TOKEN_KEY, newToken),
        AsyncStorage.setItem(USER_ID_KEY, newUserId.toString()),
      ]);

      if (userData) {
        await AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(userData));
        setUser(userData);
      }

      setToken(newToken);
      setUserId(newUserId);
      setIsAuthenticated(true);

      // 获取用户信息
      const response = await userApi.info({ xUserId: newUserId });
      if (response.data) {
        setUser(response.data);
        await AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(response.data));
      }
    } catch (error) {
      console.error('Failed to login:', error);
      throw error;
    }
  }

  async function loginWithCredentials(email: string, password: string): Promise<LoginResponseDTO> {
    try {
      const response = await userApi.signIn({ email, password });
      if (response.data && response.data.token && response.data.user) {
        const { token: newToken, user: userData } = response.data;
        const uid = userData.id ?? 0;
        
        await Promise.all([
          AsyncStorage.setItem(AUTH_TOKEN_KEY, newToken),
          AsyncStorage.setItem(USER_ID_KEY, uid.toString()),
          AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(userData)),
        ]);

        setToken(newToken);
        setUserId(uid);
        setUser(userData);
        setIsAuthenticated(true);
        
        return response.data;
      }
      throw new Error('Invalid login response');
    } catch (error) {
      console.error('Failed to login with credentials:', error);
      throw error;
    }
  }

  async function loginWithSso(username: string, password: string): Promise<{ token: string; tokenHead: string }> {
    try {
      const response = await ssoApi.login({ username, password });
      const loginData = response.data;
      if (!loginData?.token || !loginData?.tokenHead) {
        throw new Error('Invalid SSO login response');
      }
      const { token: newToken, tokenHead } = loginData;
      
      const fullToken = tokenHead + newToken;
      
      const infoResponse = await ssoApi.info({ xUserId: 0 });
      let uid = 0;
      
      if (infoResponse.data) {
        const userData: UserDTO = infoResponse.data;
        uid = userData.id ?? 0;
        
        await Promise.all([
          AsyncStorage.setItem(AUTH_TOKEN_KEY, fullToken),
          AsyncStorage.setItem(USER_ID_KEY, uid.toString()),
          AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(userData)),
        ]);

        setToken(fullToken);
        setUserId(uid);
        setUser(userData);
        setIsAuthenticated(true);
      }
      
      return loginData as { token: string; tokenHead: string };
    } catch (error) {
      console.error('Failed to login with SSO:', error);
      throw error;
    }
  }

  async function register(email: string, password: string, referrerId?: number) {
    try {
      const params: { [key: string]: any } = {
        email,
        password,
        typeId: 1,
      };
      if (referrerId) {
        params.referrerId = referrerId;
      }
      
      await userApi.create({ requestBody: params });
    } catch (error) {
      console.error('Failed to register:', error);
      throw error;
    }
  }

  async function registerWithSso(username: string, password: string) {
    try {
      await ssoApi.register({ username, password });
    } catch (error) {
      console.error('Failed to register with SSO:', error);
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
        AsyncStorage.removeItem(USER_DATA_KEY),
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
        await AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(response.data));
      }
    } catch (error) {
      console.error('Failed to refresh user:', error);
      throw error;
    }
  }

  async function loginWithPhone(_phone: string, _authCode: string) {
    // TODO: 后端 API 暂不支持手机号登录
    throw new Error('手机号登录暂不支持');
    /* try {
      const response = await ssoApi.loginByPhone({ telephone: phone, authCode });
      const loginData = response.data;
      if (!loginData?.token || !loginData?.tokenHead) {
        throw new Error('Invalid phone login response');
      }
      const { token: newToken, tokenHead } = loginData;
      
      const fullToken = tokenHead + newToken;
      
      const infoResponse = await ssoApi.info({ xUserId: 0 });
      let uid = 0;
      
      if (infoResponse.data) {
        const userData: UserDTO = infoResponse.data;
        uid = userData.id ?? 0;
        
        await Promise.all([
          AsyncStorage.setItem(AUTH_TOKEN_KEY, fullToken),
          AsyncStorage.setItem(USER_ID_KEY, uid.toString()),
          AsyncStorage.setItem(USER_DATA_KEY, JSON.stringify(userData)),
        ]);

        setToken(fullToken);
        setUserId(uid);
        setUser(userData);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('Failed to login with phone:', error);
      throw error;
    } */
  }

  async function getAuthCode(_telephone: string) {
    // TODO: 后端 API 暂不支持获取验证码
    throw new Error('获取验证码暂不支持');
    /* try {
      await ssoApi.getAuthCode({ telephone });
    } catch (error) {
      console.error('Failed to get auth code:', error);
      throw error;
    } */
  }

  async function forgotPassword(_telephone: string, _password: string, _authCode: string) {
    // TODO: 后端 API 暂不支持更新密码
    throw new Error('忘记密码功能暂不支持');
    /* try {
      await ssoApi.updatePassword({ telephone, password, authCode });
    } catch (error) {
      console.error('Failed to update password:', error);
      throw error;
    } */
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
        loginWithCredentials,
        loginWithSso,
        loginWithPhone,
        getAuthCode,
        forgotPassword,
        register,
        registerWithSso,
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
