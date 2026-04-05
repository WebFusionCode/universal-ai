{
  "brand": {
    "product_name": "AutoML X",
    "design_personality": [
      "premium",
      "futuristic",
      "Awwwards-style",
      "dark-only",
      "glass-heavy",
      "precision-engineered",
      "quietly cinematic"
    ],
    "north_star": "Make complex ML workflows feel like piloting a high-end instrument panel: calm, legible, and slightly magical (orbs + glass + glow), without sacrificing data density."
  },
  "design_tokens": {
    "notes": [
      "Dark mode only. Base background must read as #050505.",
      "Gradients are decorative only (<=20% viewport). No gradients on text-heavy surfaces.",
      "Use neon green as the primary accent for success/active states; blue as secondary accent; purple only as subtle orb light (not UI gradients on components)."
    ],
    "css_custom_properties": {
      "path": "/app/frontend/src/index.css",
      "implementation": "Replace :root and .dark tokens with the following dark-only system (keep @tailwind directives).",
      "tokens": {
        "--bg-0": "#050505",
        "--bg-1": "#0A0B0D",
        "--bg-2": "#0F1115",
        "--surface": "rgba(255,255,255,0.06)",
        "--surface-2": "rgba(255,255,255,0.08)",
        "--border": "rgba(255,255,255,0.10)",
        "--border-2": "rgba(255,255,255,0.14)",
        "--text-1": "rgba(255,255,255,0.92)",
        "--text-2": "rgba(255,255,255,0.72)",
        "--text-3": "rgba(255,255,255,0.56)",
        "--primary": "#B7FF4A",
        "--primary-2": "#7CFFB2",
        "--info": "#6AA7FF",
        "--warning": "#FFCC66",
        "--danger": "#FF5C7A",
        "--ring": "rgba(183,255,74,0.45)",
        "--shadow-soft": "0 10px 30px rgba(0,0,0,0.55)",
        "--shadow-glow": "0 0 0 1px rgba(183,255,74,0.18), 0 0 24px rgba(183,255,74,0.12)",
        "--radius-sm": "12px",
        "--radius-md": "18px",
        "--radius-lg": "24px",
        "--radius-pill": "999px",
        "--blur-strong": "18px",
        "--blur-max": "26px",
        "--grid-max": "1200px"
      },
      "shadcn_hsl_mapping": {
        "background": "0 0% 2%",
        "foreground": "0 0% 96%",
        "card": "0 0% 6%",
        "card-foreground": "0 0% 96%",
        "popover": "0 0% 6%",
        "popover-foreground": "0 0% 96%",
        "primary": "78 100% 64%",
        "primary-foreground": "0 0% 6%",
        "secondary": "220 10% 14%",
        "secondary-foreground": "0 0% 96%",
        "muted": "220 10% 14%",
        "muted-foreground": "0 0% 70%",
        "accent": "220 10% 14%",
        "accent-foreground": "0 0% 96%",
        "destructive": "346 100% 68%",
        "destructive-foreground": "0 0% 96%",
        "border": "0 0% 14%",
        "input": "0 0% 14%",
        "ring": "78 100% 64%",
        "radius": "1.25rem"
      }
    },
    "tailwind_usage": {
      "backgrounds": [
        "bg-[#050505]",
        "bg-[radial-gradient(1200px_circle_at_20%_10%,rgba(183,255,74,0.10),transparent_55%),radial-gradient(900px_circle_at_80%_20%,rgba(106,167,255,0.10),transparent_55%),radial-gradient(900px_circle_at_50%_90%,rgba(168,85,247,0.08),transparent_60%)]"
      ],
      "glass_surface": "bg-white/5 backdrop-blur-[18px] border border-white/10 rounded-[24px]",
      "glass_surface_hover": "hover:border-white/15 hover:bg-white/7",
      "focus_ring": "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgba(183,255,74,0.45)] focus-visible:ring-offset-0"
    },
    "spacing": {
      "principle": "Use 2–3x more spacing than default dashboards. Prefer 24/32/40px gaps.",
      "section_padding": "py-16 md:py-24",
      "container": "mx-auto max-w-[1200px] px-4 sm:px-6 lg:px-8"
    },
    "radius": {
      "cards": "rounded-[24px]",
      "inputs": "rounded-[18px]",
      "pills": "rounded-full"
    },
    "shadows": {
      "card": "shadow-[0_10px_30px_rgba(0,0,0,0.55)]",
      "glow_on_hover": "hover:shadow-[0_0_0_1px_rgba(183,255,74,0.18),0_0_24px_rgba(183,255,74,0.12)]"
    }
  },
  "typography": {
    "font_pairing": {
      "heading": "Plus Jakarta Sans",
      "body": "Inter",
      "mono": "IBM Plex Mono (for metrics, logs, code snippets)"
    },
    "google_fonts": {
      "add_to": "/app/frontend/public/index.html",
      "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap"
    },
    "scale": {
      "h1": "text-4xl sm:text-5xl lg:text-6xl tracking-[-0.02em] font-semibold",
      "h2": "text-base md:text-lg text-white/70",
      "h3": "text-lg md:text-xl font-semibold",
      "body": "text-sm md:text-base text-white/72 leading-relaxed",
      "small": "text-xs text-white/55"
    },
    "usage_rules": [
      "Headings: Plus Jakarta Sans with slightly tighter tracking.",
      "Body/UI: Inter for legibility in dense tables.",
      "Logs/metrics: IBM Plex Mono; keep line-height tight (leading-5)."
    ]
  },
  "layout": {
    "global": {
      "app_background": "Always render a base #050505 with subtle radial orbs behind content.",
      "no_centered_app": "Do not center-align the entire app container. Use left-aligned reading flow.",
      "grid": {
        "landing": "12-col grid on desktop; bento cards span 3–8 cols; stack on mobile.",
        "dashboard": "Fixed 100vh shell: sidebar (280px) + main content with internal ScrollArea."
      }
    },
    "landing_page_skeleton": [
      "Floating pill navbar (glass) with logo, links, auth CTA",
      "Hero: left copy + right 'builder preview' glass panel",
      "Marquee: logos / capabilities",
      "Bento features grid",
      "How it works: 3–5 steps with timeline",
      "Builder preview: dataset → train → leaderboard mock",
      "CTA band (solid, not gradient-heavy)",
      "Footer (solid bg, subtle border)"
    ],
    "auth_pages_skeleton": [
      "Full-bleed cinematic background: animated mesh + orbs",
      "Centered auth card (but page content not globally centered elsewhere)",
      "Email/password form + Google button",
      "Security microcopy + terms"
    ],
    "dashboard_shell": {
      "structure": [
        "Left: glass sidebar with nav items + workspace switcher",
        "Top (main): compact header row with breadcrumbs + actions",
        "Main: resizable panels (dataset preview / training logs)"
      ],
      "scrolling": "Only main content scrolls (internal). Sidebar stays fixed. Use ScrollArea component.",
      "resizable": "Use shadcn Resizable for dataset preview vs training logs split."
    }
  },
  "components": {
    "component_path": {
      "shadcn_primary": "/app/frontend/src/components/ui/",
      "must_use": [
        "button.jsx",
        "card.jsx",
        "input.jsx",
        "textarea.jsx",
        "tabs.jsx",
        "table.jsx",
        "badge.jsx",
        "progress.jsx",
        "scroll-area.jsx",
        "sheet.jsx",
        "dialog.jsx",
        "dropdown-menu.jsx",
        "separator.jsx",
        "tooltip.jsx",
        "sonner.jsx",
        "skeleton.jsx",
        "resizable.jsx",
        "calendar.jsx"
      ]
    },
    "navbar": {
      "style": "Floating pill glass navbar with blur and subtle border; sits over hero.",
      "classes": "sticky top-4 z-50 mx-auto w-[min(1100px,calc(100%-2rem))] rounded-full bg-white/5 backdrop-blur-[18px] border border-white/10 px-3 py-2 shadow-[0_10px_30px_rgba(0,0,0,0.55)]",
      "micro_interactions": [
        "On scroll: reduce blur slightly and add border opacity (Framer Motion).",
        "Nav links: underline appears as a 2px neon line that slides in (not text gradient)."
      ],
      "testids": {
        "nav": "top-nav",
        "login": "top-nav-login-button",
        "signup": "top-nav-signup-button"
      }
    },
    "buttons": {
      "variants": {
        "primary": {
          "look": "Neon-lime solid with dark text; subtle glow on hover.",
          "classes": "bg-[#B7FF4A] text-black hover:bg-[#C8FF73] focus-visible:ring-2 focus-visible:ring-[rgba(183,255,74,0.45)]",
          "motion": "hover: translateY(-1px) + glow shadow; active: scale(0.98)"
        },
        "secondary": {
          "look": "Glass button with border; becomes brighter on hover.",
          "classes": "bg-white/5 text-white border border-white/10 hover:bg-white/8 hover:border-white/15",
          "motion": "hover: subtle lift; active: scale(0.99)"
        },
        "ghost": {
          "look": "Text-only with neon underline on hover.",
          "classes": "bg-transparent text-white/80 hover:text-white"
        },
        "danger": {
          "look": "Solid danger for destructive actions.",
          "classes": "bg-[#FF5C7A] text-black hover:bg-[#FF7A92]"
        }
      },
      "sizes": {
        "sm": "h-9 px-3 text-sm rounded-[999px]",
        "md": "h-10 px-4 text-sm rounded-[999px]",
        "lg": "h-12 px-5 text-base rounded-[999px]"
      },
      "rule": "No gradients on buttons (keep premium + readable). Glow is allowed."
    },
    "cards_and_glass": {
      "default": "rounded-[24px] bg-white/5 backdrop-blur-[18px] border border-white/10 shadow-[0_10px_30px_rgba(0,0,0,0.55)]",
      "hover": "hover:bg-white/7 hover:border-white/15 hover:shadow-[0_0_0_1px_rgba(183,255,74,0.18),0_0_24px_rgba(183,255,74,0.12)]",
      "inner_padding": "p-5 md:p-6",
      "header": "flex items-start justify-between gap-4",
      "testid_rule": "If card contains key info (e.g., current dataset name), add data-testid on the value element."
    },
    "bento_grid": {
      "layout": "grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-6",
      "card_spans": [
        "Feature highlight: md:col-span-7",
        "Secondary: md:col-span-5",
        "Small stats: md:col-span-4"
      ],
      "content": "Each bento card has: icon (lucide), title, 1–2 lines, and a tiny preview (sparkline / chips)."
    },
    "tables": {
      "use": "shadcn Table for dataset preview, experiments, leaderboard.",
      "density": "Use compact rows: h-10; sticky header; zebra via subtle bg-white/3.",
      "classes": {
        "table_wrap": "rounded-[24px] border border-white/10 bg-white/4 backdrop-blur-[18px] overflow-hidden",
        "thead": "bg-white/6",
        "row_hover": "hover:bg-white/6"
      },
      "leaderboard": {
        "visual": "Rank column with Badge; top-1 gets neon outline; movement arrows via lucide.",
        "testids": {
          "table": "leaderboard-table",
          "row": "leaderboard-row"
        }
      }
    },
    "upload_dropzone": {
      "visual": "Large dashed glass panel with subtle animated border shimmer.",
      "classes": "rounded-[24px] border border-dashed border-white/15 bg-white/4 backdrop-blur-[18px] p-8",
      "states": {
        "idle": "text-white/70",
        "drag_over": "border-[rgba(183,255,74,0.45)] bg-white/6 shadow-[0_0_0_1px_rgba(183,255,74,0.18),0_0_24px_rgba(183,255,74,0.12)]",
        "error": "border-[rgba(255,92,122,0.45)]"
      },
      "testids": {
        "dropzone": "dataset-upload-dropzone",
        "file_input": "dataset-upload-file-input"
      }
    },
    "progress_and_logs": {
      "progress": "Use shadcn Progress with neon fill; show ETA + stage chips.",
      "logs": "Use ScrollArea with mono font; each log line has timestamp + level badge.",
      "testids": {
        "progress": "training-progress-bar",
        "logs": "training-live-logs"
      }
    },
    "chat_assistant": {
      "pattern": "Floating assistant button (bottom-right) opens Sheet/Drawer with chat.",
      "sheet": "Use shadcn Sheet for desktop; Drawer for mobile.",
      "styling": "Glass panel with strong blur; messages in rounded bubbles; assistant bubble has subtle neon border.",
      "testids": {
        "open": "ai-assistant-open-button",
        "input": "ai-assistant-message-input",
        "send": "ai-assistant-send-button"
      }
    },
    "marquee": {
      "pattern": "Infinite horizontal marquee for capabilities/logos.",
      "implementation": "CSS keyframes translateX; pause on hover; duplicate content for seamless loop.",
      "classes": "relative overflow-hidden border-y border-white/10 bg-white/3",
      "testids": {
        "marquee": "landing-marquee"
      }
    },
    "forms": {
      "use": "shadcn Form + Input + Label + Checkbox.",
      "inputs": "Glass inputs: bg-white/5 border-white/10 rounded-[18px] placeholder:text-white/35",
      "validation": "Inline error text in danger color; also toast via sonner for submit errors.",
      "testids": {
        "email": "auth-email-input",
        "password": "auth-password-input",
        "submit": "auth-submit-button",
        "google": "auth-google-button"
      }
    }
  },
  "motion": {
    "library": "framer-motion",
    "principles": [
      "Slow, premium easing: use easeOut with longer durations (0.45–0.8s).",
      "Prefer opacity + y transforms; avoid excessive scaling.",
      "Respect prefers-reduced-motion: disable orb animation + marquee speed reduction."
    ],
    "page_transitions": {
      "enter": "initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{duration:0.6,ease:[0.16,1,0.3,1]}}",
      "scroll_reveal": "Use whileInView with viewport={{once:true, amount:0.25}}"
    },
    "micro_interactions": {
      "buttons": "hover: translateY(-1px); active: scale(0.98)",
      "cards": "hover: border brightens + glow shadow",
      "table_rows": "hover highlight only (no transform)"
    },
    "background_orbs": {
      "rule": "Orbs are background-only and must not exceed 20% of viewport coverage in intensity.",
      "implementation_hint": "Use 2–3 absolutely positioned divs with radial-gradient backgrounds and animate translate/scale slowly (20–40s)."
    }
  },
  "data_viz": {
    "libraries": [
      {
        "name": "recharts",
        "use_cases": [
          "training metric over time",
          "leaderboard score distribution",
          "dataset column type breakdown"
        ],
        "install": "npm i recharts",
        "styling": "Use muted gridlines (white/10), neon green for primary series, blue for secondary."
      }
    ],
    "empty_states": {
      "pattern": "Glass card with icon + 1 sentence + primary CTA.",
      "copy_tone": "confident, minimal, technical"
    }
  },
  "accessibility": {
    "contrast": [
      "Body text should be at least white/70 on #050505.",
      "Neon green text should be used sparingly; prefer it for accents, not paragraphs."
    ],
    "focus": "All interactive elements must have visible focus ring using --ring.",
    "keyboard": "All dialogs/sheets/menus must be keyboard navigable (shadcn defaults).",
    "reduced_motion": "Provide reduced motion mode for marquee + orb animations."
  },
  "testing": {
    "data_testid_rules": [
      "Every button, link, input, select, tab trigger, dialog open/close, upload dropzone, and key info label/value must include data-testid.",
      "Use kebab-case describing role: e.g., data-testid=\"train-start-button\", not \"green-button\".",
      "For table rows: add data-testid on row container with stable id suffix (e.g., experiment-row-<id>)."
    ]
  },
  "image_urls": {
    "background_orb_textures": [
      {
        "url": "https://images.unsplash.com/photo-1707209856577-eeea3627f8bf?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA2MDV8MHwxfHNlYXJjaHwzfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwb3JiJTIwZGFyayUyMGJhY2tncm91bmR8ZW58MHx8fHRlYWx8MTc3NTM0MjczNnww&ixlib=rb-4.1.0&q=85",
        "category": "landing/auth background",
        "description": "Soft teal orb blur texture; use as optional overlay image with low opacity (0.08–0.14)."
      },
      {
        "url": "https://images.unsplash.com/photo-1708305729900-906f34a7d49d?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA2MDV8MHwxfHNlYXJjaHwxfHxhYnN0cmFjdCUyMGdyYWRpZW50JTIwb3JiJTIwZGFyayUyMGJhY2tncm91bmR8ZW58MHx8fHRlYWx8MTc3NTM0MjczNnww&ixlib=rb-4.1.0&q=85",
        "category": "dashboard background",
        "description": "Blue/green blur; good behind dashboard shell at very low opacity."
      },
      {
        "url": "https://images.unsplash.com/photo-1657894825744-1da6d5fbf24d?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA2OTV8MHwxfHNlYXJjaHwyfHxmdXR1cmlzdGljJTIwZGFyayUyMHRlY2glMjBhYnN0cmFjdCUyMGJhY2tncm91bmR8ZW58MHx8fGJsdWV8MTc3NTM0Mjc0MXww&ixlib=rb-4.1.0&q=85",
        "category": "hero decorative",
        "description": "Futuristic tech texture; use as masked corner overlay (mix-blend: screen) at opacity 0.06."
      }
    ],
    "product_mock_screens": [
      {
        "url": "https://images.unsplash.com/photo-1649682892309-e10e0b7cd40b?crop=entropy&cs=srgb&fm=jpg&ixid=M3w4NjA2OTV8MHwxfHNlYXJjaHwzfHxmdXR1cmlzdGljJTIwZGFyayUyMHRlY2glMjBhYnN0cmFjdCUyMGJhY2tncm91bmR8ZW58MHx8fGJsdWV8MTc3NTM0Mjc0MXww&ixlib=rb-4.1.0&q=85",
        "category": "builder preview placeholder",
        "description": "Abstract blue tech image; can sit behind a glass 'preview' panel until real screenshots exist."
      }
    ]
  },
  "instructions_to_main_agent": {
    "global_css_cleanup": [
      "Remove CRA demo styles from /app/frontend/src/App.css (App-logo, App-header).",
      "Set body background to #050505 and apply font-family Inter globally; headings use Plus Jakarta Sans via utility classes.",
      "Implement dark-only shadcn tokens in index.css; ensure .dark class is applied at root (or set tokens directly on :root and remove light theme usage)."
    ],
    "orb_background_scaffold_js": {
      "where": "Create a reusable <OrbBackground /> component used on Landing + Auth pages.",
      "how": "Use 3 absolutely positioned divs with radial-gradient backgrounds; animate with Framer Motion (slow drift). Ensure reduced motion fallback.",
      "do_not": [
        "Do not put gradients on cards/tables.",
        "Do not exceed 20% viewport gradient intensity."
      ]
    },
    "dashboard_shell": [
      "Implement a 100vh layout with Sidebar + Main; Main uses ScrollArea for internal scrolling.",
      "Use Resizable panels for dataset preview vs logs.",
      "All interactive elements must include data-testid attributes."
    ],
    "auth_cinematic": [
      "Use animated mesh/orbs behind a single glass auth card.",
      "Google button is secondary glass; primary submit is neon-lime."
    ]
  }
}

---

<General UI UX Design Guidelines>  
    - You must **not** apply universal transition. Eg: `transition: all`. This results in breaking transforms. Always add transitions for specific interactive elements like button, input excluding transforms
    - You must **not** center align the app container, ie do not add `.App { text-align: center; }` in the css file. This disrupts the human natural reading flow of text
   - NEVER: use AI assistant Emoji characters like`🤖🧠💭💡🔮🎯📚🎭🎬🎪🎉🎊🎁🎀🎂🍰🎈🎨🎰💰💵💳🏦💎🪙💸🤑📊📈📉💹🔢🏆🥇 etc for icons. Always use **FontAwesome cdn** or **lucid-react** library already installed in the package.json

 **GRADIENT RESTRICTION RULE**
NEVER use dark/saturated gradient combos (e.g., purple/pink) on any UI element.  Prohibited gradients: blue-500 to purple 600, purple 500 to pink-500, green-500 to blue-500, red to pink etc
NEVER use dark gradients for logo, testimonial, footer etc
NEVER let gradients cover more than 20% of the viewport.
NEVER apply gradients to text-heavy content or reading areas.
NEVER use gradients on small UI elements (<100px width).
NEVER stack multiple gradient layers in the same viewport.

**ENFORCEMENT RULE:**
    • Id gradient area exceeds 20% of viewport OR affects readability, **THEN** use solid colors

**How and where to use:**
   • Section backgrounds (not content backgrounds)
   • Hero section header content. Eg: dark to light to dark color
   • Decorative overlays and accent elements only
   • Hero section with 2-3 mild color
   • Gradients creation can be done for any angle say horizontal, vertical or diagonal

- For AI chat, voice application, **do not use purple color. Use color like light green, ocean blue, peach orange etc**

</Font Guidelines>

- Every interaction needs micro-animations - hover states, transitions, parallax effects, and entrance animations. Static = dead. 
   
- Use 2-3x more spacing than feels comfortable. Cramped designs look cheap.

- Subtle grain textures, noise overlays, custom cursors, selection states, and loading animations: separates good from extraordinary.
   
- Before generating UI, infer the visual style from the problem statement (palette, contrast, mood, motion) and immediately instantiate it by setting global design tokens (primary, secondary/accent, background, foreground, ring, state colors), rather than relying on any library defaults. Don't make the background dark as a default step, always understand problem first and define colors accordingly
    Eg: - if it implies playful/energetic, choose a colorful scheme
           - if it implies monochrome/minimal, choose a black–white/neutral scheme

**Component Reuse:**
	- Prioritize using pre-existing components from src/components/ui when applicable
	- Create new components that match the style and conventions of existing components when needed
	- Examine existing components to understand the project's component patterns before creating new ones

**IMPORTANT**: Do not use HTML based component like dropdown, calendar, toast etc. You **MUST** always use `/app/frontend/src/components/ui/ ` only as a primary components as these are modern and stylish component

**Best Practices:**
	- Use Shadcn/UI as the primary component library for consistency and accessibility
	- Import path: ./components/[component-name]

**Export Conventions:**
	- Components MUST use named exports (export const ComponentName = ...)
	- Pages MUST use default exports (export default function PageName() {...})

**Toasts:**
  - Use `sonner` for toasts"
  - Sonner component are located in `/app/src/components/ui/sonner.tsx`

Use 2–4 color gradients, subtle textures/noise overlays, or CSS-based noise to avoid flat visuals.
</General UI UX Design Guidelines>
