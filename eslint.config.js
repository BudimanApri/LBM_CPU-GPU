import { defineConfig } from 'eslint/config';
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import eslintConfigPrettier from 'eslint-config-prettier';
import globals from 'globals';

export default defineConfig(
  { ignores: ['dist/**', 'node_modules/**', 'src/gpu/shaders/generated/**'] },
  js.configs.recommended,
  tseslint.configs.strictTypeChecked,
  tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: globals.browser,
    },
    rules: {
      // noUncheckedIndexedAccess (tsconfig) forces `!` after bounds-known-safe
      // D2Q9/lattice index math -- this rule would fight that pattern everywhere.
      '@typescript-eslint/no-non-null-assertion': 'off',
      // WGSL codegen interpolates array lengths / numeric constants into
      // template strings -- numbers stringify predictably, allow them.
      '@typescript-eslint/restrict-template-expressions': ['error', { allowNumber: true }],
    },
  },
  {
    files: ['vite.config.ts', 'scripts/**/*.ts'],
    languageOptions: { globals: globals.node },
  },
  eslintConfigPrettier,
);
