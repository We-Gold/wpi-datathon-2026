import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { APP_NAME } from "./src/config.ts";

/** Fills %APP_NAME% in index.html from the same constant the app uses. */
function appName(): Plugin {
	return {
		name: "app-name",
		transformIndexHtml: (html) => html.replaceAll("%APP_NAME%", APP_NAME),
	};
}

export default defineConfig({
	plugins: [react(), appName()],
});
