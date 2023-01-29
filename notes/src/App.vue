<template lang="pug">
  div.container
    h1 {{ name }}
    span.sec-title {{ data.welcomeTitle }}
    files2video
</template>
<script setup lang="ts">
import {ref, reactive, watch, watchEffect, onMounted} from "vue"
import Files2video from "./features/Files2video.vue"

enum Status {
  NoInit,
  Loading,
  Loaded,
  End,
  Error
}

interface States {
  welcomeTitle: string
  status: Status
}

const name = ref<string>("VideoCron")
const data = reactive<States>({
  welcomeTitle: `welcome to use ${name}`,
  status: Status.NoInit
})
/*const stop = watchEffect(async onCleanup => {
  console.log("data.status change: " + data.status)
  onCleanup(() => {
    console.log("do clean")
  })
})*/
watch(() => data.status, (newVal, oldVal, onCleanup) => {
  //  newVal === oldVal
})
onMounted(() => {
  data.status = Status.Loaded
  //  只需要监听一次，之后停止监听
  // stop()
})
</script>
<style scoped>
</style>
