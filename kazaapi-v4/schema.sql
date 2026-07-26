-- =============================================================================
-- KazaAPI v4 — Supabase schema
-- Run this in the Supabase SQL editor (Project -> SQL Editor -> New query).
-- Designed to stay well within the 500MB free-tier storage budget:
--   - short-lived tables (temporary_chats) are aggressively purged,
--   - long_term_memories stores compacted TEXT summaries, not raw transcripts,
--   - JSONB is used sparingly (user_preferences only) to avoid schema bloat.
-- =============================================================================

create extension if not exists "uuid-ossp";
create extension if not exists pg_cron;  -- may require enabling in Database > Extensions

-- -----------------------------------------------------------------------------
-- temporary_chats: sliding-window raw message log, purged on compaction
-- and by the 7-day auto-purge job.
-- -----------------------------------------------------------------------------
create table if not exists temporary_chats (
    id          uuid primary key default uuid_generate_v4(),
    session_id  text not null,
    sender_role text not null check (sender_role in ('user', 'assistant', 'system')),
    content     text not null,
    created_at  timestamptz not null default now()
);

create index if not exists idx_temporary_chats_session_id on temporary_chats (session_id);
create index if not exists idx_temporary_chats_created_at  on temporary_chats (created_at);
create index if not exists idx_temporary_chats_sender_role on temporary_chats (sender_role);

-- -----------------------------------------------------------------------------
-- long_term_memories: one compacted summary per session (upserted), plus a
-- special session_id 'uri-self-learning' used by the proactive idle engine.
-- -----------------------------------------------------------------------------
create table if not exists long_term_memories (
    session_id text primary key,
    summary    text not null,
    updated_at timestamptz not null default now()
);

create index if not exists idx_long_term_memories_updated_at on long_term_memories (updated_at);

-- -----------------------------------------------------------------------------
-- user_preferences: single-row-per-user config (JSONB), API keys stored as
-- Fernet ciphertext strings (see utils/crypto.py) — never plaintext.
-- -----------------------------------------------------------------------------
create table if not exists user_preferences (
    user_key    text primary key,           -- e.g. 'kaza'
    personality jsonb not null default '{}'::jsonb,
    tool_toggles jsonb not null default '{}'::jsonb,
    api_keys    jsonb not null default '{}'::jsonb,  -- values are encrypted strings
    updated_at  timestamptz not null default now()
);

-- -----------------------------------------------------------------------------
-- flashcards: SM-2 spaced repetition state
-- -----------------------------------------------------------------------------
create table if not exists flashcards (
    id             uuid primary key default uuid_generate_v4(),
    owner_role     text not null default 'kaza',
    deck           text not null,
    front          text not null,
    back           text not null,
    repetitions    integer not null default 0,
    ease_factor    real not null default 2.5,
    interval_days  integer not null default 0,
    due_at         timestamptz not null default now(),
    created_at     timestamptz not null default now()
);

create index if not exists idx_flashcards_deck   on flashcards (deck);
create index if not exists idx_flashcards_due_at on flashcards (due_at);

-- -----------------------------------------------------------------------------
-- 7-day auto-purge function + pg_cron schedule
-- (Backstop: services/memory.py's run_auto_purge() also does this from the
-- Python side via the /api/v1/cron endpoint, in case pg_cron isn't
-- available on your Supabase plan/project.)
-- -----------------------------------------------------------------------------
create or replace function purge_old_records() returns void as $$
begin
    delete from temporary_chats where created_at < now() - interval '7 days';
    delete from long_term_memories where updated_at < now() - interval '7 days'
        and session_id <> 'uri-self-learning';  -- keep Uri's own accumulated knowledge
end;
$$ language plpgsql;

-- Schedule the purge to run daily at 03:00 UTC. If pg_cron isn't enabled on
-- your project, skip this line and rely on the Python-side purge instead.
select cron.schedule('purge_old_records_daily', '0 3 * * *', $$select purge_old_records();$$);
