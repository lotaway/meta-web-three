import React, { useState, useEffect, useRef, useCallback } from 'react'
import { View, Text, FlatList, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native'
import { Stack, useLocalSearchParams } from 'expo-router'
import { useTranslation } from 'react-i18next'
import { useConversation, CsMessage } from '@/hooks/useConversation'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

export default function ChatScreen() {
  const { sessionId } = useLocalSearchParams<{ sessionId: string }>()
  const { t } = useTranslation()
  const colorScheme = useColorScheme()
  const colors = Colors[colorScheme ?? 'light']
  const { fetchMessages, sendMessage, closeConversation } = useConversation()
  const { connected, send, on } = useWebSocket(sessionId || null)

  const [messages, setMessages] = useState<CsMessage[]>([])
  const [inputText, setInputText] = useState('')
  const [agentName, setAgentName] = useState('')
  const flatListRef = useRef<FlatList>(null)

  useEffect(() => {
    if (!sessionId) return
    fetchMessages(sessionId).then(setMessages)
  }, [sessionId, fetchMessages])

  useEffect(() => {
    const unsub = on('NEW_MESSAGE', (data: any) => {
      const msg: CsMessage = {
        id: data.messageId || Date.now().toString(),
        sessionId: sessionId || '',
        messageId: data.messageId || '',
        senderType: data.senderType || 'AGENT',
        senderId: data.senderId || 0,
        msgType: data.msgType || 'TEXT',
        content: data.content || '',
        timestamp: data.timestamp || new Date().toISOString(),
        readStatus: false,
      }
      setMessages((prev) => [...prev, msg])
    })
    const unsub2 = on('SESSION_ASSIGNED', (data: any) => {
      setAgentName(data.agentName || '')
    })
    const unsub3 = on('SYSTEM', (data: any) => {
      setMessages((prev) => [...prev, {
        id: Date.now().toString(), sessionId: sessionId || '',
        messageId: '', senderType: 'SYSTEM', senderId: 0,
        msgType: 'SYSTEM', content: data.content || '',
        timestamp: new Date().toISOString(), readStatus: false,
      }])
    })
    return () => { unsub(); unsub2(); unsub3() }
  }, [on, sessionId])

  useEffect(() => {
    flatListRef.current?.scrollToEnd({ animated: true })
  }, [messages])

  const handleSend = useCallback(async () => {
    if (!inputText.trim() || !sessionId) return
    const text = inputText.trim()
    setInputText('')
    const msg = await sendMessage(sessionId, 'TEXT', text)
    if (msg) {
      setMessages((prev) => [...prev, msg])
      send({ type: 'SEND_MESSAGE', sessionId, msgType: 'TEXT', content: text })
    }
  }, [inputText, sessionId, sendMessage, send])

  const handleClose = useCallback(async () => {
    if (!sessionId) return
    await closeConversation(sessionId)
    send({ type: 'CLOSE_SESSION', sessionId })
  }, [sessionId, closeConversation, send])

  const handleRate = useCallback(async (score: number) => {
    if (!sessionId) return
    send({ type: 'RATING', sessionId, score })
  }, [sessionId, send])

  const renderMessage = useCallback(({ item }: { item: CsMessage }) => {
    const isCustomer = item.senderType === 'CUSTOMER'
    const isSystem = item.senderType === 'SYSTEM'
    if (isSystem) {
      return (
        <View style={styles.systemRow}>
          <Text style={styles.systemText}>{item.content}</Text>
        </View>
      )
    }
    return (
      <View style={[styles.msgRow, isCustomer ? styles.msgRight : styles.msgLeft]}>
        <View style={[styles.bubble, {
          backgroundColor: isCustomer ? colors.tint : (colors.card || '#fff'),
        }]}>
          <Text style={{ color: isCustomer ? '#fff' : colors.text }}>{item.content}</Text>
          <Text style={[styles.msgTime, {
            color: isCustomer ? 'rgba(255,255,255,0.7)' : (colors.icon || '#999'),
          }]}>
            {new Date(item.timestamp).toLocaleTimeString()}
          </Text>
        </View>
      </View>
    )
  }, [colors])

  return (
    <KeyboardAvoidingView
      style={[styles.container, { backgroundColor: colors.background }]}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <Stack.Screen options={{
        title: agentName || (t('cs.customer_service') || '客服'),
        headerRight: () => (
          <TouchableOpacity onPress={handleClose}>
            <Text style={{ color: colors.tint, marginRight: 12 }}>{t('cs.close') || '结束'}</Text>
          </TouchableOpacity>
        ),
      }} />
      <View style={styles.statusBar}>
        <View style={[styles.dot, { backgroundColor: connected ? '#52c41a' : '#ff4d4f' }]} />
        <Text style={{ fontSize: 12, color: colors.icon || '#999' }}>
          {connected ? (t('cs.connected') || '已连接') : (t('cs.disconnected') || '未连接')}
        </Text>
      </View>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(item) => item.messageId || item.id}
        contentContainerStyle={styles.listContent}
        onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })}
      />
      <View style={[styles.inputBar, { borderTopColor: colors.border || '#eee' }]}>
        <TextInput
          style={[styles.input, {
            backgroundColor: colors.card || '#fff',
            color: colors.text,
            borderColor: colors.border || '#ddd',
          }]}
          value={inputText}
          onChangeText={setInputText}
          placeholder={t('cs.input_placeholder') || '输入消息...'}
          multiline
        />
        <TouchableOpacity style={[styles.sendBtn, { backgroundColor: colors.tint }]} onPress={handleSend}>
          <Text style={styles.sendText}>{t('common.send') || '发送'}</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  )
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  statusBar: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 6, gap: 6 },
  dot: { width: 8, height: 8, borderRadius: 4 },
  listContent: { padding: 16 },
  msgRow: { marginBottom: 12, flexDirection: 'row' },
  msgLeft: { justifyContent: 'flex-start' },
  msgRight: { justifyContent: 'flex-end' },
  bubble: { maxWidth: '75%', padding: 10, borderRadius: 12, },
  msgTime: { fontSize: 10, marginTop: 4, textAlign: 'right' },
  systemRow: { alignItems: 'center', marginVertical: 8 },
  systemText: { fontSize: 12, color: '#999', fontStyle: 'italic' },
  inputBar: { flexDirection: 'row', padding: 12, borderTopWidth: StyleSheet.hairlineWidth, alignItems: 'flex-end' },
  input: { flex: 1, borderWidth: 1, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8, maxHeight: 100 },
  sendBtn: { marginLeft: 8, paddingHorizontal: 16, paddingVertical: 10, borderRadius: 8, },
  sendText: { color: '#fff', fontWeight: '500' },
})
