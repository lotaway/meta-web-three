<template lang="pug">
.file2video
  ul.list.list-select
    li.item.item-select(v-for="(s, sIndex) in selectors" :key="s.title")
      directory-selector(:title="s.title" :button-title="buttonTitle" :placeholder="s.placeholder" @pathChange="filePaths => pathChangeHandler(sIndex, filePaths)")
  button.btn.btn-generate(@click="generateVideo")
</template>
<style scoped>

</style>
<script lang="ts" setup>
import {ipcRenderer} from "electron"
import {ref, onMounted} from "vue"
import DirectorySelector from "../components/DirectorySelector.vue"

interface SelectorVal {
  title: string,
  desc: string,
  placeholder?: string,
  directoryPath: string
}

type SelectorKey = "general" | "widget" | "platform"
type Selector = Record<SelectorKey, SelectorVal>
// type Props = { buttonTitle?: string } & Selector
interface Props extends Selector {
  buttonTitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  general: () => ({
    title: "General Folder:",
    desc: "the files in General will be used in all videos(Intro, Closer...)",
    directoryPath: ""
  }),
  widget: () => ({
    title: "Widgets Folder:",
    desc: "a folder with files related to a platform",
    directoryPath: ""
  }),
  platform: () => ({
    title: "Platform Folder:",
    desc: "a folder with sub-folders, each of them related to a different widget.",
    directoryPath: ""
  }),
  buttonTitle: "Select"
})
const selectors = ref([
  props.general,
  props.widget,
  props.platform
])

async function pathChangeHandler(index: number, directoryPath: string) {
  selectors.value[index].directoryPath = directoryPath
  // window.desktop.Files2videoPathChangeHandler(directoryPath)
  const files = await ipcRenderer.invoke("readFileInDirectory", directoryPath)
}

function generateVideo() {

}

onMounted(async () => {
  const defaultDirectory = localStorage.getItem("Files2video:Directory")
  if (!defaultDirectory) return false
})
</script>
<script lang="ts">
export default {
  name: "Files2video"
}
</script>
