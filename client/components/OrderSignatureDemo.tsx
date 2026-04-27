import { Button, StyleSheet, Text, View } from 'react-native'
import { ORDER_SIGNATURE_DEMO_PAYLOAD } from '@/features/order-signature/orderSignature.constants'
import { useOrderSignature } from '@/features/order-signature/useOrderSignature'

type Props = {
  onSignatureReady: (signature: string) => void
}

function getStatusText(
  status: 'idle' | 'loading' | 'success' | 'error',
  total: string | undefined,
  errorMessage: string | null,
) {
  if (status === 'loading') {
    return '签名中...'
  }

  if (status === 'success' && total) {
    return `总价 ${total}`
  }

  if (status === 'error' && errorMessage) {
    return `签名失败: ${errorMessage}`
  }

  return '准备签名'
}

export default function OrderSignatureDemo({ onSignatureReady }: Props) {
  const { status, errorMessage, result, signOrder } = useOrderSignature(
    ORDER_SIGNATURE_DEMO_PAYLOAD,
  )

  async function handleSignOrder() {
    const signature = await signOrder()

    if (signature) {
      onSignatureReady(signature)
    }
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>订单签名 Demo</Text>
      <Text style={styles.status}>
        {getStatusText(status, result?.total, errorMessage)}
      </Text>
      <Button title="生成签名" onPress={handleSignOrder} />
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    backgroundColor: '#f9f9f9',
    borderRadius: 8,
    margin: 10,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
  },
  status: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
  },
})
