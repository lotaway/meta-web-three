<template>
  <div class="cs-quick-reply">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>快捷回复</span>
          <el-button type="primary" size="small" @click="showDialog = true">添加</el-button>
        </div>
      </template>
      <el-table :data="list" border stripe>
        <el-table-column prop="id" label="ID" width="60" />
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="content" label="内容" show-overflow-tooltip />
        <el-table-column prop="msgType" label="类型" width="100" />
        <el-table-column prop="sort" label="排序" width="70" />
        <el-table-column prop="createTime" label="创建时间" width="170" />
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button size="small" type="danger" @click="handleDelete(row.id)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
    <el-dialog v-model="showDialog" title="添加快捷回复" width="500px">
      <el-form :model="form" label-width="80px">
        <el-form-item label="分组ID"><el-input v-model.number="form.groupId" type="number" /></el-form-item>
        <el-form-item label="标题"><el-input v-model="form.title" /></el-form-item>
        <el-form-item label="内容"><el-input v-model="form.content" type="textarea" :rows="4" /></el-form-item>
        <el-form-item label="消息类型">
          <el-select v-model="form.msgType">
            <el-option label="文本" value="TEXT" />
            <el-option label="图片" value="IMAGE" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" @click="handleCreate">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getQuickReplyListAPI, createQuickReplyAPI, deleteQuickReplyAPI } from '@/apis/cs'
import type { QuickReply } from '@/apis/cs'
import { ElMessage } from 'element-plus'

const list = ref<QuickReply[]>([])
const showDialog = ref(false)
const form = ref<QuickReply>({ id: null, groupId: null, title: '', content: '', msgType: 'TEXT', sort: 0, createTime: '' })

const load = async () => {
  const res = await getQuickReplyListAPI()
  list.value = res.data || []
}

const handleCreate = async () => {
  await createQuickReplyAPI(form.value)
  ElMessage.success('创建成功')
  showDialog.value = false
  form.value = { id: null, groupId: null, title: '', content: '', msgType: 'TEXT', sort: 0, createTime: '' }
  load()
}

const handleDelete = async (id: number) => {
  await deleteQuickReplyAPI(id)
  ElMessage.success('已删除')
  load()
}

onMounted(load)
</script>
