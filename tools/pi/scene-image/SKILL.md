---
name: scene-image
description: Render a picture of the current scene, character, place or moment from this conversation with one of the local mako-image models (mako_image_serve), and show it. Use when the user asks for an image, illustration, picture or visual of what is being written, imagined or built in the session.
---

# Scene image

Turn what the conversation has established into one image prompt, render it with mako_image_serve, and show the result.

0. Pick the image model (see "Models" below) unless one is already in use in this conversation.
1. Write the prompt as "Writing the prompt" below says.
2. Pick a size: 1024x1024, or 1344x768 for a landscape or wide scene, 768x1344 for a standing figure or tall scene
   (each side a multiple of 32).
3. Run, from this skill's directory:

   ```sh
   scripts/render.sh --model MODEL WIDTHxHEIGHT "<the prompt>"
   ```

   (Its options are under "Options".) It prints the path of the PNG it saved. It takes from a few seconds to a few
   minutes (the model's description gives its time). When the image server shares the Mac with the chat model's
   mako_serve (`--llm`), the chat model is unloaded while the image model is loaded; the next message loads it again
   (~10 s), and this conversation continues as before.
4. The script opens the image in the Mac's image viewer. Reply with its path and the prompt used, in a line or two.
   Do not `read` the image unless the user wants to talk about what is in it (it then goes to the model as image
   input, which needs the chat model's mako_serve started with `--mmproj`). If the user wants changes, revise the
   prompt as "Revising" says and render again. What the user says about a render describes that render (often a
   fault to fix), not words to add to the prompt; see "Revising".

## Writing the prompt

The image model knows only what the prompt says and fills the rest with its most common choice. Most models' text
encoders are language models, so write English sentences, not keyword lists (a model whose prompting notes say it takes
tags can have them in front; one that takes only Danbooru tags: see "Tag prompts"). It binds words to subjects loosely:
every mention of a person, every "he" or "she" and every named body part is one more chance to attach something to the
wrong body, draw a person twice or put a limb in the wrong place. A good prompt says each thing once, in one place, and
nothing the image cannot show.

Build it in this order:

1. Medium and arrangement, in one or two sentences: the style (the one the user asked for in this session, else one
   that suits the material: "a cinematic photograph", "a watercolour"), how many people when more than one ("two
   people: a man and a woman"), and where each is in what pose, by the pose's name ("she sits on the bench, he stands
   behind it").
2. One block per subject, all of it together: a fixed handle ("the tall woman", "the man in the grey coat"), then what
   the story has fixed about how it looks (age, build, hair, face, clothing piece by piece with colour and state,
   things held), then what the arrangement leaves open (hands, gaze, expression). Inside its own block "he" or "she"
   is clear; elsewhere use the handle. Use the same words for a character in every image. At most one clause on
   contact with another subject, in the block of the one doing it ("her hand on the man's shoulder").
3. The setting in particular things (walls, floor, furniture, light sources), the time of day, the weather.
4. The camera and light: the shot (wide, medium, close-up), viewpoint and height, the light's direction and colour,
   the mood.

Rules:

- Concrete and literal: say what is seen, not an abstraction ("an intimate moment", "a fantasy outfit"). Use the
  user's terms and everything the conversation fixed for what is shown, however long ago; leave out what does not
  belong in this image; do not soften or summarize what the user asked for.
- Each fact once: restating it in other words ("on her back, chest up, back on the table, torso flat") adds
  references, not weight.
- Say what is there, not what is not: "not sitting up" or "no hat" tends to put the thing in the image. What must
  stay out goes in `--negative` (see "Options"). No text or lettering unless the user wants it.
- Text to render: the exact words in quotes, in their own language and case. Where the lines must break, put line
  breaks in the quoted text (MOSS-Image follows them; otherwise it breaks lines itself).
- Only what this camera sees: from behind a person, their face and the hands in front of them are hidden; "in
  profile" and "from behind" exclude each other. Describe what shows instead (hands where they are visible, the back
  of the head).
- Ends, sides and directions that cannot be misread: "the near end of the table, closest to the camera", "the far
  end"; left and right as the viewer sees them.
- A gaze or gesture needs a target already placed ("looking at the man"); without one the model adds one.
- No quality words ("masterpiece", "8K", "highly detailed") unless the model's prompting notes ask for quality tags:
  describe what is seen. No size or aspect ratio in the text: that is the render's size.
- Length comes from the content, not from padding: a scene with several people and exact placement needs more words
  than a portrait, each still said once. Models cut long prompts: Krea 2 at about 500 tokens, Qwen-Image at 1024.

Before rendering, read the prompt back as the camera sees the scene: who is nearest, who is on the left, what each
hand touches, what is hidden. Then count: the people (stated when more than one, each introduced once) and each
person's mentions outside their own block (at most two). Remove what is said twice or negated.

## Tag prompts

For a model whose prompting notes say Danbooru-style tags (Illustrious and other SDXL anime models, Anima), the tags
must be ones the model learned: an invented tag ("sparkly dress", "long silver hair") is read as loose words or not
at all. `scripts/tags.py` looks them up in Danbooru's tag list with each tag's post count (a tag on a few dozen
images is barely learned).

- The same build order, as tags: quality and rating tags as the notes give them; the count (`1girl`, `2boys`); the
  character and series tags; per subject its look (hair, eyes, clothing piece by piece with colour), pose and
  expression; the setting; the framing and light (`upper body`, `from below`, `backlighting`). Tags lowercase with
  spaces, separated by commas.
- `scripts/tags.py search WORDS` when unsure what a thing is called: it lists the tags containing the words, most
  used first (`search ponytail`: ponytail, side ponytail, high ponytail ...); a character's tag usually carries the
  series (`search anne`: takamaki anne). Artist and meta tags only with `--all`, and an artist tag only when the user
  asks for that artist's style.
- Before rendering, `scripts/tags.py check "<prompt>"`, and rewrite what it flags: `alias` (use the name it gives),
  `unknown` (pick from its suggestions or search), `rare` (fewer than 500 posts: keep it only if nothing more common
  says it), `prose` (split into tags). Quality and rating words are conventions, not tags, and pass.
- Weights, `(tag:1.2)`, only for a tag that a render showed too weakly, 1.1 to 1.4.

The list is `~/models/tags/danbooru.csv` (`MAKO_TAGS_CSV`); if the script says there is none, write the tags as best
you can and tell the user the check did not run.

## Revising

After a render, read what the user says about it as a report on that image first:

- A plain statement about the picture ("the man is facing left", "she has three hands", "the sign says OPEN") is
  what the user sees in it, usually a fault. It is not text for the prompt: work out what the image should show
  instead (from the prompt and the conversation) and change the prompt toward that. Never paste the user's
  statement into the prompt.
- It is a request only when it is worded as one: "make", "change to", "should", "I want", "instead", "can you".
- When you cannot tell whether a statement says what the image shows or what the user wants, ask in one short
  line ("Is that what the render shows and you want it fixed, or how it should be?") and do not render until they
  answer.
- Example. Prompt: "...the man faces the woman...". User: "the man is facing left". The render has him turned away
  from her (she stands on the right): make his facing unmistakable ("the man, seen in profile, faces right, toward
  the woman"). Wrong: adding "the man is facing left".

When the user wants a change or a render came out wrong:

- Find the part that caused it (the arrangement, one subject's block, the setting, the camera) and rewrite that part
  in place; keep the others word for word. The prompt stays about as long as it was.
- Remove before adding: first take out what conflicts or competes (a second mention of someone, a detail the camera
  cannot see, a clause about another subject's body), then rephrase, and add only if that fails.
- A wrong number of people, body parts on the wrong person or merged figures usually come from too many
  cross-references: shorten.
- When the cause is unclear, change one thing per render, so that the result shows what worked.
- When several subjects keep landing in the wrong places with a model that takes layouts, use one (see "Layouts"): a
  box per person fixes where each goes and how many there are.

## Layouts

MOSS-Image models take a layout: a caption for the whole image and a box per character, object or text, which fixes
where each one goes. Use one when the placement of several subjects matters or a render put them in the wrong
places. Render a draft with `scripts/render.sh --model MODEL --layout DRAFT` (the image size comes from the draft),
DRAFT being a file, `-` to read it from standard input (e.g. a heredoc), or the JSON itself as one argument:

```json
{"width": 1344, "height": 768,
 "summary": {"caption": "<the whole image: setting, light, style, and the subjects and their arrangement in brief>"},
 "items": [
  {"label": "character", "caption": "<this one only: look, clothing, pose, expression, one contact at most>",
   "bbox": [80, 96, 640, 768]},
  {"label": "character", "caption": "<the other one>", "bbox": [600, 64, 1260, 768]},
  {"label": "object", "caption": "<a thing that matters>", "bbox": [520, 520, 820, 760]}
 ]}
```

- `bbox` is `[left, top, right, bottom]` in pixels from the image's top-left corner; width and height are multiples
  of 32. Boxes may overlap where subjects overlap or touch; a box is where the subject is, not a frame drawn around
  it.
- A box's caption is drawn inside the box, so it describes only that subject, as a block of "Writing the prompt"
  does; describing another subject's body there draws it twice. `summary.caption` gives the medium, the arrangement
  in brief and the setting.
- Labels: `character` for people and creatures, `object` for things, `text` (with `"text": "<the exact words>"`,
  `\n` where a line must break) for lettering. A text item's caption gives the lettering's look (size, weight,
  colour, the surface it is on); give separate items to text placed apart.
- Text must fit its box at the image's size: a long line or a paragraph needs a much wider box than a short title.
  Enlarge or rearrange the boxes rather than shortening the user's wording.
- Optional: `summary.meta`, a list of medium or genre words (`["poster"]`, `["product photography"]`);
  `summary.canvas`, `single` (default), `multi_panel` or `collage`; `palette` (in summary or an item), a list of
  colours. The layout is the whole prompt: put any prose in `summary.caption`.
- A layout has positions in the frame, not depth: which subject is in front, or hides part of another, goes in the
  captions ("partly behind the fern"); the order of the items did not change it when tried.

## Models

`scripts/models.sh` lists the image models the image server has: each one's name, which model it is, a description of
what it is good at, how its prompts should be written and how long it takes, and its default LoRAs; one is marked
"default" (used without `--model`) and one "loaded" (rendering with it needs no reload).

- Pick by what the user asks for and by the descriptions: the style (anime and illustration, photograph, painting),
  what the image must contain (legible text, a layout), the kind of content, and the time the user will wait. A model
  the user names wins. If none fits clearly, use the default and say which you used.
- Keep the same model for the rest of the conversation, so characters and style stay consistent; change it only when
  the user asks or the material changes (a photograph after anime illustrations).
- Write the prompt the way the chosen model's `prompting` notes say (prose, tags, prompt weights, layouts), on top
  of "Writing the prompt".
- Rendering with another model than the loaded one first loads it (a few seconds more).

## LoRAs

LoRAs are add-ons to an image model (a style, a kind of content, a faster schedule), each model with its own.
`scripts/loras.sh MODEL` lists the chosen model's, by name; those marked "default" are what every render with it
uses unless told otherwise, at the scale shown.

- Without `--lora`, a render uses the model's defaults: leave `--lora` out unless you change them. With
  `--lora NAME[:SCALE]` (repeatable), it uses exactly the LoRAs given and no others, so to add one to the defaults,
  pass the defaults too:
  `scripts/render.sh --lora fast-4step:1 --lora watercolour:0.8 1024x1024 "<the prompt>"`.
  `--no-loras` renders with none.
- SCALE is the strength (1 when left out). Where the list gives a LoRA's scales, start in the middle of that range
  and stay within it. If the user finds an effect too weak or too strong, raise or lower it by 0.2-0.3.
- A model's default LoRAs are part of how it is set up (a speed LoRA such as "dmd", "lightning", "4step", "turbo"
  goes with the low step count and guidance the model is set to); the LoRAs you added are the user's choices. When
  the user asks to remove LoRAs, remove the ones you added and keep the defaults: pass the defaults alone as
  `--lora`, or leave `--lora` out if only defaults remain. Use `--no-loras` only when the user asks to drop the
  defaults too, and then, if a default was a speed LoRA, switch to the same model's folder without it instead (its
  description says which, e.g. `moss-sft` for `moss-sft-dmd`): without it the low step count gives soft, smeared
  images.
- Use a LoRA only when the user asks for it or its name clearly matches what they want. Once the user has chosen
  LoRAs, pass the same ones on every later render in the conversation: nothing is remembered between renders.
- A LoRA listed with a trigger acts only when the prompt contains it: work the trigger into the prompt, word for word
  (at the start if it ends in a colon, like "Next Scene:", else within the sentence where it fits). A description
  says what the LoRA is for. A LoRA with neither may still need words the list does not show; tell the user so if
  its effect does not show.
- A render with other LoRAs than the previous one first reloads the image model (a few seconds more). LoRAs do not
  carry over to another model: list the new model's and choose again.

## Options

`scripts/render.sh [options] WIDTHxHEIGHT "<prompt>"`, or `scripts/render.sh [options] --layout DRAFT`:

- `--model NAME`: the image model (default: the server's default).
- `--lora NAME[:SCALE]` (repeatable): exactly these LoRAs in place of the model's defaults; `--no-loras`: none (see
  "LoRAs"). A `:merge` or `:unmerged` suffix overrides how the server adds it; leave it off unless the user asks.
- `--steps N`: denoising steps, 0 to 100; 0 is the model's own count; without it the step count the server set for the
  model applies (lower with a speed LoRA). More steps take longer. Leave it out unless the user asks for more or fewer
  steps, or the speed LoRA a model's low step count is set for is gone.
- `--negative "TEXT"`: what the image should not show, as a short list ("blurry, extra fingers, text, watermark"),
  when the user wants something kept out or a render keeps showing something unwanted. It adds a second transformer
  pass a step, so the render takes about twice as long. Qwen-Image follows it well; Krea 2 Turbo barely (it was
  trained to run without one). MOSS-Image always runs that pass (against an empty negative prompt without one), and
  Anima (not Turbo) and SDXL against a default one, so there it costs no extra time. A model at guidance 1 (the
  models list shows it; speed-LoRA setups) skips the pass, and a negative prompt has no effect.
- `--guidance F`: how strongly the prompt, against the negative prompt, steers the image (0 to 30; default the model's:
  Qwen-Image 4, Krea 2 Turbo 2, MOSS-Image 3, Anima 4, SDXL 6). Too high gives flat, oversaturated or broken images
  (Krea 2 Turbo from 3). Change it only when the user asks, or by 1 at a time when a negative prompt shows too little or
  too much.
- `--sampler euler|euler_a|er_sde`: euler_a for SDXL models only (their default; it adds fresh noise each step, so a
  seed's image changes with the step count); er_sde for any model (also adds noise each step). Leave it out unless the
  user asks.
- `--skip-blocks LIST`: MOSS-Image only, the blocks its negative prompt's pass leaves out, e.g. `22,24` (22, 24
  and 26 are the ones its makers suggest; `""` for none, which looks more photographic but can lose detail; default
  the model's, 22). Leave it out unless the user asks.
- `--seed N`: the noise's seed; the same seed and settings give the same image. Without it each render gets a new
  one (the PNG records it). Use it to compare one change at a time, or when the user wants a render repeated.
- `--layout DRAFT`: a MOSS-Image layout in place of size and prompt (see "Layouts").

The environment, set by the user: `MAKO_IMAGE_URL` and `MAKO_IMAGE_API_KEY` (the server), `MAKO_IMAGE_DIR` (where
images are saved, default `~/Pictures/mako`), `MAKO_IMAGE_OPEN=0` (do not open the image viewer), `MAKO_TAGS_CSV`
(the tag list). `scripts/models.sh` takes no options; `scripts/loras.sh [MODEL]` lists the server's default model's
LoRAs without one.

If a script fails, report its message: mako_image_serve may not be running (or not at `MAKO_IMAGE_URL`), may need an
API key (`MAKO_IMAGE_API_KEY`), or may not have a model or LoRA of that name (list them again). A message starting
"borrowing memory:" means the image server, started with `--llm URL` to share the Mac with mako_serve, could not
reach mako_serve there (not running, another port, or it wants `--llm-api-key`); the user fixes that where the image
server is started.
