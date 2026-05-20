import type { ReactNode } from 'react';
import { DarkTheme, DefaultTheme, ThemeProvider } from '@react-navigation/native';
import { useFonts } from 'expo-font';
import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { StripeProvider } from '@stripe/stripe-react-native';
import 'react-native-reanimated';
import './../i18n';

import { STRIPE_PUBLISHABLE_KEY } from '@/api/generated';
import { useColorScheme } from '../hooks/useColorScheme';
import { AuthProvider } from '@/contexts/AuthContext';

function PaymentProviders({ children }: { children: ReactNode }) {
  if (!STRIPE_PUBLISHABLE_KEY) {
    return <>{children}</>;
  }
  return (
    <StripeProvider publishableKey={STRIPE_PUBLISHABLE_KEY}>
      {children}
    </StripeProvider>
  );
}

export default function RootLayout() {
  const colorScheme = useColorScheme();
  const [loaded] = useFonts({
    SpaceMono: require('../assets/fonts/SpaceMono-Regular.ttf'),
  });

  if (!loaded) {
    // Async font loading only occurs in development.
    return null;
  }

  return (
    <AuthProvider>
      <PaymentProviders>
        <ThemeProvider value={colorScheme === 'dark' ? DarkTheme : DefaultTheme}>
          <Stack>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="+not-found" />
          </Stack>
          <StatusBar style="auto" />
        </ThemeProvider>
      </PaymentProviders>
    </AuthProvider>
  );
}
