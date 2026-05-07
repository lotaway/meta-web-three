import { Image, ImageProps } from 'expo-image'
import { useState } from 'react'
import { View, StyleSheet, ActivityIndicator } from 'react-native'

interface CachedImageProps extends Omit<ImageProps, 'source'> {
  uri: string
  placeholder?: number
  showLoading?: boolean
}

export function CachedImage({
  uri,
  placeholder,
  showLoading = true,
  style,
  ...props
}: CachedImageProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(false)

  return (
    <View style={[styles.container, style]}>
      <Image
        {...props}
        source={{ uri }}
        style={styles.image}
        contentFit="cover"
        transition={200}
        cachePolicy="memory-disk"
        onLoadStart={() => setLoading(true)}
        onLoadEnd={() => setLoading(false)}
        onError={() => {
          setError(true)
          setLoading(false)
        }}
      />
      {showLoading && loading && (
        <View style={styles.loader}>
          <ActivityIndicator size="small" color="#999" />
        </View>
      )}
      {error && placeholder && (
        <Image
          source={placeholder}
          style={styles.image}
          contentFit="cover"
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loader: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
})

export function useImagePreloader() {
  const preload = async (uris: string[]) => {
    const promises = uris.map((uri) => {
      return new Promise((resolve) => {
        Image.prefetch(uri).then(resolve).catch(resolve)
      })
    })
    await Promise.all(promises)
  }

  return { preload }
}

export function useLazyImage() {
  const [visible, setVisible] = useState(false)

  const show = () => setVisible(true)

  return { visible, show }
}