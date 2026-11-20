import React, { useState, useEffect, useRef, useCallback } from 'react'
import { View, Text, FlatList, TextInput, TouchableOpacity, StyleSheet, KeyboardAvoidingView, Platform } from 'react-native'
import { Stack, useLocalSearchParams } from 'expo-router'
import { useTranslation } from 'react-i18next'
import { useConversation, CsMessage } from '@/hooks/useConversation'
import { useWebSocket } from '@/hooks/useWebSocket'
import { Colors } from '@/constants/Colors'
import { useColorScheme } from '@/hooks/useColorScheme'

type AppColors = typeof Colors.light
type Translate = ReturnType<typeof useTranslation>['t']
type SetMessages = React.Dispatch<React.SetStateAction<CsMessage[]>>
type SetString = React.Dispatch<React.SetStateAction<string>>
type ChatOn = ReturnType<typeof useWebSocket>['on']
type ChatSend = ReturnType<typeof useWebSocket>['send']
type FlatListRef = React.RefObject<FlatList>
type RouteSessionId = string | undefined

function buildIncomingMessage(data: any, sessionId: string): CsMessage {
  return {
    id: data.messageId || Date.now().toString(),
    sessionId,
    messageId: data.messageId || '',
    senderType: data.senderType || 'AGENT',
    senderId: data.senderId || 0,
    msgType: data.msgType || 'TEXT',
    content: data.content || '',
    timestamp: data.timestamp || new Date().toISOString(),
    readStatus: false,
  }
}

function buildSystemMessage(data: any, sessionId: string): CsMessage {
  return {
    id: Date.now().toString(), sessionId,
    messageId: '', senderType: 'SYSTEM', senderId: 0,
    msgType: 'SYSTEM', content: data.content || '',
    timestamp: new Date().toISOString(), readStatus: false,
  }
}

function registerChatListeners(sessionId: string, on: ChatOn, setMessages: SetMessages, setAgentName: SetString) {
  const unsub = on('NEW_MESSAGE', (data: any) => {
    setMessages((prev) => [...prev, buildIncomingMessage(data, sessionId)])
  })
  const unsub2 = on('SESSION_ASSIGNED', (data: any) => {
    setAgentName(data.agentName || '')
  })
  const unsub3 = on('SYSTEM', (data: any) => {
    setMessages((prev) => [...prev, buildSystemMessage(data, sessionId)])
  })
  return () => { unsub(); unsub2(); unsub3() }
}

async function submitChatMessage(sessionId: string, text: string, sendMessage: ReturnType<typeof useConversation>['sendMessage'], setMessages: SetMessages, send: ChatSend) {
  const msg = await sendMessage(sessionId, 'TEXT', text)
  if (msg) {
    setMessages((prev) => [...prev, msg])
    send({ type: 'SEND_MESSAGE', sessionId, msgType: 'TEXT', content: text })
  }
}

function useChatState() {
  const [messages, setMessages] = useState<CsMessage[]>([])
  const [inputText, setInputText] = useState('')
  const [agentName, setAgentName] = useState('')
  const flatListRef = useRef<FlatList>(null)
  return { messages, setMessages, inputText, setInputText, agentName, setAgentName, flatListRef }
}

type ChatState = ReturnType<typeof useChatState>

function useChatEffects(sessionId: RouteSessionId, state: ChatState, fetchMessages: ReturnType<typeof useConversation>['fetchMessages'], on: ChatOn) {
  useEffect(() => {
    if (!sessionId) return
    fetchMessages(sessionId).then(state.setMessages)
  }, [sessionId, fetchMessages])
  useEffect(() => {
    return registerChatListeners(sessionId || '', on, state.setMessages, state.setAgentName)
  }, [on, sessionId])
  useEffect(() => {
    state.flatListRef.current?.scrollToEnd({ animated: true })
  }, [state.messages])
}

function useChatHandlers(sessionId: RouteSessionId, state: ChatState, sendMessage: ReturnType<typeof useConversation>['sendMessage'], closeConversation: ReturnType<typeof useConversation>['closeConversation'], send: ChatSend) {
  const handleSend = useCallback(async () => {
    if (!state.inputText.trim() || !sessionId) return
    const text = state.inputText.trim()
    state.setInputText('')
    await submitChatMessage(sessionId, text, sendMessage, state.setMessages, send)
  }, [state.inputText, sessionId, sendMessage, send])
  const handleClose = useCallback(async () => {
    if (!sessionId) return
    await closeConversation(sessionId)
    send({ type: 'CLOSE_SESSION', sessionId })
  }, [sessionId, closeConversation, send])
  const handleRate = useCallback(async (score: number) => {
    if (!sessionId) return
    send({ type: 'RATING', sessionId, score })
  }, [sessionId, send])
  return { handleSend, handleClose, handleRate }
}

function useChatController() {
  const { sessionId } = useLocalSearchParams<{ sessionId: string }>()
  const { t } = useTranslation()
  const colorScheme = useColorScheme()
  const colors = Colors[colorScheme ?? 'light']
  const { fetchMessages, sendMessage, closeConversation } = useConversation()
  const { connected, send, on } = useWebSocket(sessionId || null)
  const state = useChatState()
  useChatEffects(sessionId, state, fetchMessages, on)
  return { t, colors, connected, sessionId, ...state, ...useChatHandlers(sessionId, state, sendMessage, closeConversation, send) }
}

interface ChatViewProps extends ChatState, ReturnType<typeof useChatHandlers> {
  t: Translate
  colors: AppColors
  connected: boolean
}

function SystemMessageRow({ item }: { item: CsMessage }) {
  return (
    <View style={styles.systemRow}>
      <Text style={styles.systemText}>{item.content}</Text>
    </View>
  )
}

function ChatBubble({ item, colors }: { item: CsMessage; colors: AppColors }) {
  const isCustomer = item.senderType === 'CUSTOMER'
  return (
    <View style={[styles.msgRow, isCustomer ? styles.msgRight : styles.msgLeft]}>
      <View style={[styles.bubble, { backgroundColor: isCustomer ? colors.tint : (colors.card || '#fff') }]}>
        <Text style={{ color: isCustomer ? '#fff' : colors.text }}>{item.content}</Text>
        <Text style={[styles.msgTime, { color: isCustomer ? 'rgba(255,255,255,0.7)' : (colors.icon || '#999') }]}>
          {new Date(item.timestamp).toLocaleTimeString()}
        </Text>
      </View>
    </View>
  )
}

function MessageItem({ item, colors }: { item: CsMessage; colors: AppColors }) {
  if (item.senderType === 'SYSTEM') return <SystemMessageRow item={item} />
  return <ChatBubble item={item} colors={colors} />
}

function StatusBadge({ connected, colors, t }: Pick<ChatViewProps, 'connected' | 'colors' | 't'>) {
  return (
    <View style={styles.statusBar}>
      <View style={[styles.dot, { backgroundColor: connected ? '#52c41a' : '#ff4d4f' }]} />
      <Text style={{ fontSize: 12, color: colors.icon || '#999' }}>
        {connected ? (t('cs.connected') || '已连接') : (t('cs.disconnected') || '未连接')}
      </Text>
    </View>
  )
}

function ChatInputBar({ colors, t, inputText, setInputText, handleSend }: Pick<ChatViewProps, 'colors' | 't' | 'inputText' | 'setInputText' | 'handleSend'>) {
  return (
    <View style={[styles.inputBar, { borderTopColor: colors.border || '#eee' }]}>
      <TextInput
        style={[styles.input, { backgroundColor: colors.card || '#fff', color: colors.text, borderColor: colors.border || '#ddd' }]}
        value={inputText} onChangeText={setInputText}
        placeholder={t('cs.input_placeholder') || '输入消息...'} multiline
      />
      <TouchableOpacity style={[styles.sendBtn, { backgroundColor: colors.tint }]} onPress={handleSend}>
        <Text style={styles.sendText}>{t('common.send') || '发送'}</Text>
      </TouchableOpacity>
    </View>
  )
}

function ChatView(props: ChatViewProps) {
  const { colors, t, connected, agentName, messages, inputText, setInputText, flatListRef, handleClose, handleSend } = props
  return (
    <KeyboardAvoidingView style={[styles.container, { backgroundColor: colors.background }]} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <Stack.Screen options={{
        title: agentName || (t('cs.customer_service') || '客服'),
        headerRight: () => (
          <TouchableOpacity onPress={handleClose}>
            <Text style={{ color: colors.tint, marginRight: 12 }}>{t('cs.close') || '结束'}</Text>
          </TouchableOpacity>
        ),
      }} />
      <StatusBadge connected={connected} colors={colors} t={t} />
      <FlatList ref={flatListRef} data={messages} renderItem={({ item }) => <MessageItem item={item} colors={colors} />} keyExtractor={(item) => item.messageId || item.id} contentContainerStyle={styles.listContent} onContentSizeChange={() => flatListRef.current?.scrollToEnd({ animated: true })} />
      <ChatInputBar colors={colors} t={t} inputText={inputText} setInputText={setInputText} handleSend={handleSend} />
    </KeyboardAvoidingView>
  )
}

export default function ChatScreen() {
  const controller = useChatController()
  return <ChatView {...controller} />
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
