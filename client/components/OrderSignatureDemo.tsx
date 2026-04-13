import React, { useState } from 'react'
import { View, Text, Button, StyleSheet } from 'react-native'
import Appsdk from 'react-native-appsdk'

interface Props {
    onSignatureReady: (signature: string) => void
}

export default function OrderSignatureBeta({ onSignatureReady }: Props) {
    const [status, setStatus] = useState('准备签名')

    const handleSignOrder = async () => {
        try {
            setStatus('签名中...')
            const nonce = await Appsdk.createNonce()
            const ts = await Appsdk.systemTimestampMs()

            const orderData = {
                buyerId: 'U12345',
                sku: 'META001',
                quantity: 1,
                nonce,
                timestampMs: ts,
            }

            const total = await Appsdk.computeOrderTotal('299.00', 1, '0.00', '15.00')
            const signature = await Appsdk.generateRequestSignature(orderData, 'demo-key')

            setStatus(`总价 ${total}`)
            onSignatureReady(signature)
        } catch (error: any) {
            setStatus(`签名失败: ${error.message}`)
        }

        return (
            <View style={styles.container}>
                订单签名 Beta
                <Text style={styles.status}>{status}</Text>
                <Button title="生成签名" onPress={handleSignOrder} />
            </View>
        )
    }
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

