// This file must be a module (not ambient) to correctly augment vue-router.
// The empty export ensures TypeScript treats it as a module.
export {};

declare module "vue-router" {
  interface RouteMeta {
    fullscreen?: boolean;
  }
}
