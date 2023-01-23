<template>
  <ul class="list list-select">
    <li v-for="s in selector" class="item item-select" :key="s.title">
      <directory-selector :title="s.title" :button-title="buttonTitle" :placeholder="s.placeholder"
                          @pathChange="pathChangeHandler"/>
    </li>
  </ul>
</template>
<style scoped>

</style>
<script lang="ts" setup>
//  todo @require("./files2video.pdf") 根据文档完成文件输出成视频的需求
import fs from "fs"
import path from "path"
import {onMounted} from "vue"
import DirectorySelector from "../components/DirectorySelector.vue"
function pathChangeHandler(filePath: string) {
  const generaPath = path.join(__dirname, "./General")
  const platformPath = path.join(__dirname, "./Platform")
  const widgetsPath = path.join(__dirname, "./Widgets")
  const destinationFolder = path.join(__dirname, "./build")
  const filenames = getIncludeFiles(filePath)
  const filePaths = filename2path(filenames, filePath)
}

function getIncludeFiles(path: string): string[] {
  const items = fs.readdirSync(path, {
    withFileTypes: true
  })
  if (items.length === 0) return []
  let filenames: string[] = []
  items.forEach((item, index) => item.isFile() && filenames.push(item.name))
  return filenames
}

function filename2path(filenames: string[], prevFix: string) {
  return filenames.map(filename => path.join(prevFix, filename))
}

interface Props {
  selector?: Array<{
    title: string,
    placeholder?: string,
    directoryPath: string
  }>
  buttonTitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  selector: () => [
    {
      title: "General Folder:",
      directoryPath: ""
    },
    {
      title: "Widgets Folder:",
      directoryPath: ""
    },
    {
      title: "Platform Folder:",
      directoryPath: ""
    }
  ],
  buttonTitle: "Select"
})

onMounted(() => {

})
</script>
<script lang="ts">
export default {
  name: "Files2video"
}
</script>
