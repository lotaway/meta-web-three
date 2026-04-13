import React, { useState } from 'react'
import { View, Text, Button, Alert, StyleSheet } from 'react-native'
import Appsdk from 'react-native-appsdk'

export default function PasskeyAuthBeta() {
    const [status, setStatus] = useState('就绪')

    const handleAuth = async () => {
        try {
            setStatus('验证中')
            const challenge = Appsdk.generateChallenge()
            const verified = await Appsdk.authenticatePasskey(challenge)
            setStatus(verified ? '验证成功' : '验证失败')
        } catch (error: any) {
            Alert.alert('验证失败', error.message)
            setStatus('验证失败')
        }
    }

    return (
        <View style={styles.container}>
            <Text style={styles.title}>设备指纹 Beta</Text>
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
        fontWeight: '600',
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

