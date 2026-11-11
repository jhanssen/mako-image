---
name: edit-image
description: Make a new image from one or more reference images and a prompt with a local image model that takes references (MiniMax-H3-Image, Qwen-Image 2.1) through mako_image_serve. Use it whenever the user gives, names or points at an image file (a photo, an earlier render, a screenshot, a 3D blockout, a product shot) that the new image should keep something from, edit a picture (change its background, weather, time of day, clothing, pose, text, colours), put a person, animal, object or character from one picture into another scene, keep a character or product consistent across renders, combine several pictures into one (furniture into a room, portraits into a group), or turn a render or sketch into a photograph, even when the user says none of "reference", "edit" or "image to image".
---

# Edit image

Turn one or more reference images and what the user wants changed, kept or combined into one prompt, render it
through mako_image_serve with the references attached, and look at the result.

1. Pick the model: `scripts/models.sh` lists them; use one whose entry says it takes reference images, and keep
   the same one for the rest of the conversation. Two do: minimax-h3-image (up to 16 references, a described
   target image as its prompt, 6 steps; the default here) and qwen-image (its card: up to 10, an instruction as
   its prompt, 40 steps; also the one to use when the user names it or wants its look). The entry's `prompting`
   notes are the model's own; this file applies them.
2. Read each reference (the Read tool shows the image). Write down what the new image keeps from it: each subject
   as it actually looks (sex, age, build, hair, face, clothing piece by piece, colours, things held), the setting,
   the camera's position and the light. The prompt defines each kept element from what the picture shows, and a
   definition that disagrees with the picture ("the man in a blue coat" when it is green) makes the model redraw it
   from the words instead of taking it from the picture.
3. Write the prompt as "The prompt" says below.
4. Pick the size: the aspect ratio of the reference that is kept whole, sides multiples of 32. The model sees a
   reference with its short edge brought within 1024 to 2048 pixels (a 1344x768 picture at 1792x1024, a 2048x1024
   one as it is, a 6000x4000 photo at 3072x2048). By default the model renders on a canvas of at least 2048x2048
   pixels' area at the asked ratio and resizes to the size (about 95 s with one reference); `--min-area 0` with the
   reference's own size renders at that size (about 70 s at 2048x1024), which is where edits were checked. Sizes up
   to 3072x1024 and 2944x1472 came out clean; larger ones are untested.
5. Run, from this skill's directory, the references in the order the prompt numbers them:

   ```sh
   scripts/render.sh --model minimax-h3-image --reference A.png --reference B.png 2048x1024 "<the prompt>"
   ```

   It prints the path of the PNG it saved (`~/Pictures/mako`, `MAKO_IMAGE_DIR`) and opens it in the image viewer
   (`MAKO_IMAGE_OPEN=0` not to). About 70 s for 2048x1024 with one reference of that size and `--min-area 0`
   (6 steps of 6 s, the reference's encoding, the text encoder), about 95 s on the default canvas; each further
   reference adds its encoding and its tokens to every step; the first render of a session loads the model (5 s
   when its files are in the page cache, a few minutes from the SSD).
6. Read the result and compare it with the references and the request: is each preserved subject the same one
   (face, hair, clothing, colours, proportions), did the asked change happen, did anything else drift (the setting,
   the camera, text). Reply with the path, what the render kept and changed, and anything it got wrong, in a few
   lines. When it is wrong, revise as "Revising" says and render again.

## The prompt

For qwen-image the prompt is an instruction about the pictures, as to a person: "Change the weather to a clear
summer evening at sunset, the sun low on the left; keep the woman, the lighthouse and the gulls exactly as they
are." Say what changes and what stays; the template names the pictures `<image1>`, `<image2>`, ... for the model,
so "the first image" or "the woman in the first image" is enough. Pass `--guidance 1` (its pipeline edits without
guidance) and a size at the main reference's aspect ratio near 1024x1024 pixels' area (1440x736 for a 2:1
picture; its pipeline's default), about 85 s with one reference at 40 steps. Checked once: a sunset edit of a
2048x1024 render kept its subjects and composition and changed the weather.

MiniMax-H3-Image takes one English paragraph describing the whole finished image, used as given: no template, no
tags, no quality words, each fact once, said concretely (what is seen, not "a cozy mood"). The references are
`<Picture 1>`, `<Picture 2>`, ... in the order passed. An edit prompt has three parts, in this order, in one
paragraph (the model's own prompt expander writes them this way):

1. **Definitions.** One sentence per element the new image keeps or borrows: "The man in a dark sweater standing at
   the counter in `<Picture 1>` is `<Subject 1>`." A picture can define several subjects, or supply only a colour,
   texture or style ("The teal of the door in `<Picture 2>` is `<Subject 2>`"). Describe the element as the picture
   shows it, enough to single it out, no more.
2. **The target.** "The target image shows ..." in one sentence for the whole picture, then one clause per subject
   saying how much of it survives: "`<Subject 1>` is fully preserved." / "`<Subject 1>` is partially preserved;
   identity, hair and clothing remain, while the pose changes to sitting on the bench with both hands on the
   backrest." / "`<Subject 2>` supplies the exact colour transferred to the car's paint."
3. **What changes or is new**, in full detail: the new setting or object, where it touches the kept subjects, the
   shadows it casts and receives, the perspective, the light's direction and colour, the medium ("photograph, 50 mm
   lens", "oil painting"). Never re-describe a preserved subject here: every word about it is a word the model may
   redraw from instead of copying the picture.

Then the general rules: state camera position, lens or medium and each body's orientation to the camera; a gaze or
gesture needs a target already placed; say what is there, not what is not; text to render in single quotes with
where it sits; about 250 words for a scene, less for a simple edit.

Examples:

- Change the surroundings, keep the subject: "The woman in a red raincoat standing at the end of the pier in
  `<Picture 1>` is `<Subject 1>`. The target image shows the same pier at night in heavy snowfall. `<Subject 1>` is
  fully preserved. The sky is blue-black with no moon; large snowflakes fall, lit from the left by the pier's
  sodium lamps, which lay warm pools of light on the snow-covered planks; the water beyond the railing is dark and
  still, with the lamps' reflections broken by small waves. The camera position, lens and framing are those of
  `<Picture 1>`; photograph."
- Combine two references: "The grey tabby cat sitting upright in `<Picture 1>` is `<Subject 1>`. The green velvet
  armchair in `<Picture 2>` is `<Subject 2>`. The target image shows `<Subject 1>` asleep on the seat of
  `<Subject 2>` in a sunlit living room. `<Subject 1>` is partially preserved; its markings, colouring and face
  remain, while the pose changes to lying curled with its tail over its nose. `<Subject 2>` is fully preserved. The
  chair stands at a three-quarter angle to the camera on a pale oak floor, a tall window on the right; afternoon sun
  comes through the window from the right, lighting the cat's back and the chair's right arm and laying a soft
  shadow on the floor to the left. Medium shot from seat height, 50 mm lens, photograph."
- A render into a photograph: "The untextured grey 3D render of a kitchen in `<Picture 1>` is `<Subject 1>`. The
  target image shows the same kitchen as a photograph. `<Subject 1>` is partially preserved; the layout, the camera
  position and every object's placement remain, while the materials and light become real: white painted cabinets
  with brass handles, a honed black granite counter, a pale oak floor, morning light through the window on the left
  falling across the counter, the room otherwise lit by its warm ceiling spots; 28 mm lens, photograph."

## What the model does with a reference

- Up to 16 references (MiniMax-H3-Image; qwen-image's card says 10), each at least 32 pixels a side and at most
  16 megapixels, in any format macOS reads (PNG, JPEG, HEIC, ...). MiniMax-H3-Image's text encoder sees each at a
  1024-pixel short edge and its VAE copies it at a 1024 to 2048-pixel short edge; qwen-image takes each at about
  1024x1024 pixels' area at its aspect ratio. A small crop is enlarged and a large photo reduced: detail finer than
  that is not kept, and a reference above about 2 megapixels buys nothing over one at 2.
- Each reference adds its tokens to every step: more or larger references, slower renders. Pass only the pictures
  the prompt defines a subject from; crop a reference to the part that matters when the rest is irrelevant.
- The model keeps what the prompt says to keep and redraws the rest. Without a definition and a preservation
  clause an element from a reference may be taken, changed or dropped.
- The result is a new image: lighting, grain and colour balance can shift over the whole picture, and a "fully
  preserved" subject is a close copy, not pixels.

## Revising

Read what the user says about a render as a report on that image first: "her coat is blue now" says the render
changed something, and the fix is in the prompt (a stronger preservation clause, a definition that matches the
picture), not that sentence pasted in. A request is worded as one ("make", "change", "should", "instead").

- A preserved subject drifted (another face, other clothing): check the definition against the picture and shorten
  it; use "fully preserved" where the pose did not need to change; move anything said about that subject in part 3
  out.
- The change did not happen: say it in part 3 in more concrete, visible terms (what replaces what, where the light
  comes from, what it touches), and check it does not contradict a preservation clause.
- Something came along from a reference that was not wanted (its background, a second person): define only what is
  borrowed ("the lamp on the left of `<Picture 2>` is `<Subject 2>`") and describe the new setting in part 3.
- Composition shifted: give the camera position, height and lens, and the aspect ratio of the source as the size.
- A render that is a posterized copy of the reference (its texture turned into blobs and cut-outs, rain streaks
  into white drops) is the noise, not the prompt: one seed at one size did this on every prompt tried while other
  seeds at that size and the same seed at other sizes were clean. Render again with another seed.
- Keep the seed (`--seed N`, the PNG records it) and change one thing per render when the cause is unclear, so the
  result shows what the change did.

## Options

`scripts/render.sh [options] --reference FILE... WIDTHxHEIGHT "<prompt>"`:

- `--model NAME`: the image model (the list marks the default and the loaded one; another than the loaded one is
  loaded first).
- `--seed N`: the noise's seed; the same seed, references and prompt give the same image.
- `--steps N`: the model's own count is 6; 12 is crisper at twice the time, 3 softer at half. Leave it out unless
  the user asks.
- `--min-area N`: the smallest canvas in pixels the model renders on (the server's default 4194304, 2048x2048,
  then resized to the size asked for); `0` renders at the size itself, faster for an edit at the reference's size
  (step 4).
- `--upscale NAME[:SCALE]`: an ESRGAN upscaler after the render (`scripts/upscalers.sh` lists them, with what each
  is for; photographic ones for photographs, anime ones for drawn images). Adds about a second.
- `--lora NAME[:SCALE]`: a LoRA by name from `scripts/loras.sh MODEL` in place of the model's defaults; only when
  the user asks for one. MiniMax-H3-Image takes no negative prompt and no guidance scale.
- `--guidance 1` for qwen-image edits (above); its `--steps` default is 40 (fewer is faster and softer).
- Without `--reference`, the same script renders from the prompt alone (then write a plain description of the
  whole image, no `<Picture N>`).

The environment: `MAKO_IMAGE_URL` (default `http://127.0.0.1:8082/v1`), `MAKO_IMAGE_API_KEY` (when the server wants
one), `MAKO_IMAGE_DIR`, `MAKO_IMAGE_OPEN`. If a script fails, report its message: mako_image_serve may not be
running, may want an API key, or may not have a model of that name (`scripts/models.sh` again). A reference the
server refuses (too small, too large, unreadable) is named in the error with its index.
