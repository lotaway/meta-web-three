<template>
  <label :for="title">{{ props.title }}</label>
  <input type="text" readonly value="" :id="title" :placeholder="props.placeholder"/>
  <button type="button" @click="buttonHandler">{{ props.buttonTitle }}</button>
</template>
<script setup lang="ts">
// import {defineProps, withDefaults} from "vue"
import {dialog} from "@electron/remote"

const props = withDefaults(defineProps<{
  title?: string,
  buttonTitle?: string,
  placeholder?: string
}>(), {
  title: "",
  buttonTitle: "选择",
  placeholder: ""
})
const emits = defineEmits(["buttonClick", "pathChange"])

function buttonHandler() {
  console.log("buttonHandler")
  emits("buttonClick")
  dialog.showOpenDialog({
    title: "请选择文件夹",
    properties: ["openDirectory"]
  }).then(result => {
    if (result.canceled) return false
    alert(JSON.stringify(result))
    emits("pathChange", result.filePaths)
  })

}
</script>
<script lang="ts">
export default {
  name: "DirectorySelector"
}
</script>

<style scoped>

</style>
