import React, { useState } from 'react'
import { View, Text, Button, Alert, StyleSheet } from 'react-native'
import Appsdk, { generateChallenge } from 'react-native-appsdk'

interface Props {
    onVerified: (isVerified: boolean) => void
}

PasskeyAuthBeta
const [status, setStatus] = useState('就绪')

const handleAuth = async () => {
    try {
        setStatus('验证中...')
        const challenge = generateChallenge()
        const verified = await Appsdk.authenticatePasskey(challenge)
        setStatus(verified ? '验证成功' : '验证失败')
        onVerified(verified)
    } catch (error: any) {
        Alert.alert('验证失败', error.message)
        setStatus('验证失败')
    }
}

return (
    <View style={styles.container}>
        设备指纹 Beta
        <Text style={styles.status}>{status}</Text>
        <Button title="生物识别登录" onPress={handleAuth} />
    </View>
)
}

const styles = StyleSheet.create({
    container: {
        padding: 20,
    },
    title: {
        fontSize: 18,
        fontWeight: 'bold',
        marginBottom: 10,
    },
    status: {
        fontSize: 16,
        color: '#666',
        marginBottom: 20,
        padding: 10,
        backgroundColor: '#f5f5f5',
        borderRadius: 5,
    },
})

