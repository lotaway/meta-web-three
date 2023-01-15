# Nodejs Web Socket

要注意虽然Nodejs的net模块也可以创建TCP/WebSocket服务，但一般都是通过封装更好的第三方模块完成实际业务。
本例子选用nodejs-websocket模块，也可以选择像是ws、socket.io等第三方包。
下面将创建多人聊天室，让game1客户端发送信息到game2客户端中。

## 服务端

先在后端程序里创建一个WebSocket服务端监听：

```javascript
const websocketModule = require("nodejs-websocket")
const userData = [
    {
        id: 1,
        nickname: "player1"
    },
    {
        id: 2,
        nickname: "player2"
    }
].map(item => ({
    ...item,
    isReady: item.isRead ?? false
}))
let messageList = []
const server = websocketModule.createServer(connect => {
    const messageHandler = msg => {
        switch (msg.type) {
            case "init":
                const curIndex = userData.findIndex(item => item.id === msg.id)
                userData[curIndex].isReady = true
                break;
            case "userMsg":
                messageList.push(msg)
                connect.send(result)
            default:
                break;
        }
        if (msg.type === "init") {
        }
        messageList.push(msg)
    }
    connect.on("text", msgStr => {
        const msg = JSON.parse(msgStr)
        console.log("收到的信息是：" + msgStr)
        const result = messageHandler(msg)
    })
    connect.on("close", (code, reason) => {
        console.log("关闭链接")
    })
    connect.on("error", (code, reason) => {
        console.log("异常关闭")
    })
})
server.listen(8001)
```

## 客户端脚本

```javascript
//  @web-socket-lib
let userId = null

export function createWebSocket(userInfo, receiver) {
    const webSocket = new WebSocket('ws://localhost:8001')
    webSocket.onopen = event => {
        console.log('服务已经成功连接')
        userId = userInfo.id
        webSocket.send(JSON.stringify({
            id: userInfo.id,
            type: "init",
            msg: `${userInfo.nickname}已进入聊天室`
        }))
        // 关闭Socket....
        //socket.close()
    }
    webSocket.onmessage = message => {
        receiver(message)
    }
    return {
        send(message) {
            webSocket.send({
                ...message,
                id: userId
            })
        }
    }
}
```

# 客户端调用脚本

在前端客户端，用户player1，player2访问时，调用脚本为他创建一个WebSocket连接到我们已经创建好的服务上。

## 客户端player1

```javascript
import {createWebSocket} from "web-socket-lib"

const webSocket = createWebSocket({
    id: 1
}, data => {
    console.log('客户端1接收到信息：', data)
})
```

## 客户端player2

```javascript
import {createWebSocket} from "web-socket-lib"

const webSocket = createWebSocket({
    id: 2
}, data => {
    console.log('客户端2接收到信息：', data)
})
webSocket.send({
    msg: "大家好"
})
```

用户链接完成后，player2会发送一条消息