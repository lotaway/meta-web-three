<script setup lang="ts">
import {ref} from "vue"
import Files2video from "./features/Files2video.vue"

const welcomeTitle = ref<string>("VideoCron")
</script>
<template>
  <h1>{{ welcomeTitle }}</h1>
  <Files2video/>
</template>

<style scoped>
</style>
