import { Tabs } from 'expo-router';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Platform } from 'react-native';

import { HapticTab } from '../../components/HapticTab';
import { IconSymbol } from '../../components/ui/IconSymbol';
import TabBarBackground from '../../components/ui/TabBarBackground';
import { Colors } from '../../constants/Colors';
import { useColorScheme } from '../../hooks/useColorScheme';

function createTabScreenOptions(colorScheme: 'light' | 'dark') {
  return {
    tabBarActiveTintColor: Colors[colorScheme].tint,
    headerShown: false,
    tabBarButton: HapticTab,
    tabBarBackground: TabBarBackground,
    tabBarStyle: Platform.select({
      ios: {
        // Use a transparent background on iOS to show the blur effect
        position: 'absolute' as const,
      },
      default: {},
    }),
  };
}

function TabScreen({ name, title, iconName }: {
  name: string;
  title: string;
  iconName: React.ComponentProps<typeof IconSymbol>['name'];
}) {
  return (
    <Tabs.Screen
      name={name}
      options={{
        title,
        tabBarIcon: ({ color }) => <IconSymbol size={28} name={iconName} color={color} />,
      }}
    />
  );
}

export default function TabLayout() {
  const { t } = useTranslation();
  const colorScheme = useColorScheme();

  return (
    <Tabs screenOptions={createTabScreenOptions(colorScheme ?? 'light')}>
      <TabScreen name="index" title={t('common.tabs.home')} iconName="house.fill" />
      <TabScreen name="category" title={t('common.tabs.category')} iconName="squareshape.fill" />
      <TabScreen name="cart" title={t('common.tabs.cart')} iconName="cart.fill" />
      <TabScreen name="messages" title={t('common.tabs.messages') || '消息'} iconName="message.fill" />
      <TabScreen name="profile" title={t('common.tabs.profile')} iconName="person.fill" />
    </Tabs>
  );
}
