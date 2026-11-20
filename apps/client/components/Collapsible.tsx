import { PropsWithChildren, useState } from 'react';
import { StyleSheet, TouchableOpacity } from 'react-native';

import { ThemedText } from './ThemedText';
import { ThemedView } from './ThemedView';
import { IconSymbol } from './ui/IconSymbol';
import { Colors } from '../constants/Colors';
import { useColorScheme } from '../hooks/useColorScheme';

function CollapsibleHeader({ title, isOpen, onPress, iconColor }: {
  title: string;
  isOpen: boolean;
  onPress: () => void;
  iconColor: string;
}) {
  return (
    <TouchableOpacity
      style={styles.heading}
      onPress={onPress}
      activeOpacity={0.8}>
      <IconSymbol
        name="chevron.right"
        size={18}
        weight="medium"
        color={iconColor}
        style={{ transform: [{ rotate: isOpen ? '90deg' : '0deg' }] }}
      />

      <ThemedText type="defaultSemiBold">{title}</ThemedText>
    </TouchableOpacity>
  );
}

export function Collapsible({ children, title }: PropsWithChildren & { title: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const theme = useColorScheme() ?? 'light';

  return (
    <ThemedView>
      <CollapsibleHeader
        title={title}
        isOpen={isOpen}
        onPress={() => setIsOpen((value) => !value)}
        iconColor={theme === 'light' ? Colors.light.icon : Colors.dark.icon}
      />
      {isOpen && <ThemedView style={styles.content}>{children}</ThemedView>}
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  heading: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  content: {
    marginTop: 6,
    marginLeft: 24,
  },
});
