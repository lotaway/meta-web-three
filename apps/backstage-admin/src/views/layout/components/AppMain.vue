<script lang="ts" setup>
// Define component name
defineOptions({
  name: 'AppMain'
})
</script>

<template>
  <section class="app-main">
    <router-view v-slot="{ Component }">
      <transition name="fade" mode="out-in">
        <component :is="Component" />
      </transition>
    </router-view>
  </section>
</template>
