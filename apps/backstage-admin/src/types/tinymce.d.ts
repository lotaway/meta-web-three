// TinyMCE ships a single root type declaration but no types for the deep
// side-effect imports (plugins, icons, themes, models, skins) this project
// uses to bundle the editor through Vite.
declare module 'tinymce/*'

// The translation bundles only register data on the global TinyMCE instance.
declare module 'tinymce-i18n/*'
